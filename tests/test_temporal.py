import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from loss import dice_loss, focal_loss
from model import AGDSN, TemporalState
from train import evaluate_one_epoch, slice_subclip_targets, train_one_epoch


torch.set_num_threads(1)


def small_model(**kwargs):
    options = dict(d_model=12, nhead=2, num_encoder_layers=1,
                   num_decoder_layers=1, frame_memory_len=3,
                   clip_memory_len=7, prediction_len=5)
    options.update(kwargs)
    return AGDSN(**options).eval()


class OffsetInjection(nn.Module):
    def forward(self, tgt, memory):
        return tgt + 100


class TemporalContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_independent_capacities_and_history_normalization(self):
        model = small_model()
        self.assertEqual((model.frame_memory_len, model.clip_memory_len, model.prediction_len), (3, 7, 5))
        initial = model.initial_state(2)
        self.assertEqual(initial.frame_memory.shape, (2, 3, 12))
        self.assertEqual(initial.clip_memory.shape, (2, 7, 12))
        self.assertEqual(initial.frame_memory.count_nonzero(), 0)
        self.assertEqual(initial.clip_memory.count_nonzero(), 0)
        frame_history = torch.randn(2, 5, 12, dtype=torch.float64)
        long_history = torch.randn(2, 2, 12, dtype=torch.float64)
        captured = {}

        def capture_present(module, args, kwargs):
            captured['present_memory'] = kwargs['memory']

        def capture_context(module, args, kwargs):
            captured['context_queries'] = kwargs['tgt']

        handles = [
            model.present_injection.register_forward_pre_hook(capture_present, with_kwargs=True),
            model.context_encoder.register_forward_pre_hook(capture_context, with_kwargs=True),
        ]
        output = model(torch.randn(2, 1, 12), TemporalState(frame_history, long_history))
        for handle in handles:
            handle.remove()
        expected_recent = frame_history[:, -3:].float()
        positions = torch.arange(3).float()[:, None]
        rates = torch.exp(torch.arange(0, 12, 2).float() * (-torch.log(torch.tensor(10000.0)) / 12))
        encoding = torch.zeros(3, 12)
        encoding[:, 0::2] = torch.sin(positions * rates)
        encoding[:, 1::2] = torch.cos(positions * rates)
        torch.testing.assert_close(captured['present_memory'][:, :3], expected_recent + encoding)
        expected_long = torch.cat([torch.zeros(2, 5, 12), long_history.float()], dim=1)
        torch.testing.assert_close(captured['context_queries'], expected_long)
        torch.testing.assert_close(output['state'].frame_memory[:, :2], expected_recent[:, 1:])
        torch.testing.assert_close(output['state'].clip_memory, torch.cat([expected_long[:, 1:], expected_recent[:, :1]], dim=1))
        self.assertEqual(output['state'].frame_memory.dtype, torch.float32)
        self.assertEqual(output['state'].clip_memory.device, output['current_logits'].device)

    def test_zero_capacities_stay_empty_and_attention_is_finite(self):
        for history_len, long_len in [(0, 0), (0, 4), (2, 0)]:
            with self.subTest(history=history_len, long=long_len):
                model = small_model(frame_memory_len=history_len, clip_memory_len=long_len)
                model.observation_encoder = nn.Identity()
                observed = torch.randn(2, history_len + 1, 12)
                output = model.forward_window(observed)
                self.assertEqual(output['state'].frame_memory.shape, (2, history_len, 12))
                self.assertEqual(output['state'].clip_memory.shape, (2, long_len, 12))
                self.assertTrue(torch.isfinite(output['current_logits']).all())
                self.assertTrue(torch.isfinite(output['future_logits']).all())
                (output['current_logits'].square().mean() + output['future_logits'].square().mean()).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                if history_len == 0 and long_len:
                    torch.testing.assert_close(output['state'].clip_memory[:, -1:], observed)

    def test_recent_padding_and_older_truncation(self):
        model = small_model()
        model.observation_encoder = nn.Identity()
        current = torch.randn(2, 1, 12)
        short_recent = torch.randn(2, 1, 12)
        long_older = torch.randn(2, 9, 12)
        output = model(current, TemporalState(short_recent, long_older))
        expected_recent = torch.cat([torch.zeros_like(current), short_recent, current], dim=1)
        expected_older = torch.cat([long_older[:, -6:], torch.zeros_like(current)], dim=1)
        torch.testing.assert_close(output['state'].frame_memory, expected_recent)
        torch.testing.assert_close(output['state'].clip_memory, expected_older)

    def test_current_loss_reaches_same_step_temporal_path(self):
        model = small_model()
        captured = {}

        def capture_output(name):
            def hook(module, args, output):
                captured[name] = output
                output.retain_grad()
            return hook

        def capture_inputs(name):
            def hook(module, args, kwargs):
                captured[name] = kwargs
            return hook

        handles = [
            model.present_injection.register_forward_hook(capture_output('y')),
            model.context_encoder.register_forward_hook(capture_output('context')),
            model.anticipation_decoder.register_forward_hook(capture_output('anticipated')),
            model.context_encoder.register_forward_pre_hook(capture_inputs('context_inputs'), with_kwargs=True),
            model.anticipation_decoder.register_forward_pre_hook(capture_inputs('anticipation_inputs'), with_kwargs=True),
            model.observation_decoder.register_forward_pre_hook(capture_inputs('injection_inputs'), with_kwargs=True),
        ]
        observed = torch.randn(2, 4, 12, requires_grad=True)
        output = model.forward_window(observed)
        targets = torch.randint(0, 4, (2, 1))
        current_loss = focal_loss(output['current_logits'], targets) + dice_loss(output['current_logits'], targets)
        current_loss.backward()
        for handle in handles:
            handle.remove()
        self.assertIs(captured['context_inputs']['memory'], captured['y'])
        self.assertIs(captured['anticipation_inputs']['tgt'], captured['y'])
        self.assertIs(captured['anticipation_inputs']['memory'], captured['context'])
        self.assertIs(captured['injection_inputs']['memory'], captured['anticipated'])
        for key in ['y', 'context', 'anticipated']:
            self.assertTrue(torch.isfinite(captured[key].grad).all())
            self.assertGreater(captured[key].grad.abs().sum().item(), 0)
        for module in [model.observation_encoder, model.present_injection, model.context_encoder,
                       model.anticipation_decoder, model.observation_decoder]:
            total = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
            self.assertGreater(total, 0)
        self.assertGreater(observed.grad[:, :-1].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.clip_classifier.parameters()))
        for history in output['state']:
            self.assertFalse(history.requires_grad)
            self.assertIsNone(history.grad_fn)

    def test_history_stores_chronological_pre_injection_z(self):
        model = small_model(frame_memory_len=2, clip_memory_len=3)
        model.observation_encoder = nn.Identity()
        model.observation_decoder = OffsetInjection()
        state = None
        seen = []
        for time_index in range(1, 8):
            current = torch.full((1, 1, 12), float(time_index))
            previous = state
            output = model(current, state)
            state = output['state']
            seen.append(current)
            padded = [torch.zeros_like(current)] * 5 + seen
            expected = torch.cat(padded[-5:], dim=1)
            torch.testing.assert_close(state.frame_memory, expected[:, -2:])
            torch.testing.assert_close(state.clip_memory, expected[:, :3])
            if previous is not None:
                self.assertEqual(previous.frame_memory[0, -1, 0].item(), time_index - 1)
        self.assertLess(state.frame_memory.max().item(), 100)

    def test_reset_and_independent_video_states(self):
        model = small_model()
        first = torch.randn(1, 1, 12)
        other_video = torch.randn(1, 1, 12)
        first_output = model(first)
        saved_state = tuple(tensor.clone() for tensor in first_output['state'])
        model(other_video)
        reset_output = model(first)
        explicit_reset = model(first, model.initial_state(1))
        for key in ['current_logits', 'future_logits']:
            torch.testing.assert_close(reset_output[key], first_output[key])
            torch.testing.assert_close(explicit_reset[key], first_output[key])
        for before, after in zip(saved_state, first_output['state']):
            torch.testing.assert_close(before, after)

    def test_window_matches_stream_with_correct_initial_long_history(self):
        model = small_model(frame_memory_len=3, clip_memory_len=5, prediction_len=2)
        features = torch.randn(2, 9, 12)
        state = None
        with torch.no_grad():
            for frame in features[:, :-1].split(1, dim=1):
                state = model(frame, state)['state']
            streamed = model(features[:, -1:], state)
            windowed = model.forward_window(features[:, -4:], long_history=state.clip_memory)
        for key in ['current_logits', 'future_logits']:
            torch.testing.assert_close(windowed[key], streamed[key], rtol=1e-5, atol=1e-6)
        for window_history, stream_history in zip(windowed['state'], streamed['state']):
            torch.testing.assert_close(window_history, stream_history, rtol=1e-5, atol=1e-6)

    def test_backbone_window_accepts_noncontiguous_video_slice(self):
        extractor = nn.Sequential(nn.Flatten(), nn.Linear(12, 384))
        with patch('model.timm.create_model', return_value=extractor):
            model = small_model(use_backbone=True)
        video = torch.randn(2, 7, 3, 2, 2)
        observed = video[:, :model.frame_memory_len + 1]
        self.assertFalse(observed.is_contiguous())
        with torch.no_grad():
            sliced = model.forward_window(observed)
            contiguous = model.forward_window(observed.contiguous())
        for key in ['current_logits', 'future_logits']:
            torch.testing.assert_close(sliced[key], contiguous[key], rtol=0, atol=0)
        for sliced_history, contiguous_history in zip(sliced['state'], contiguous['state']):
            torch.testing.assert_close(sliced_history, contiguous_history, rtol=0, atol=0)

    def test_current_inclusive_labels_and_independent_prediction_length(self):
        labels = torch.arange(64).unsqueeze(0)
        current, future = slice_subclip_targets(labels)
        torch.testing.assert_close(current, labels[:, 32:33])
        torch.testing.assert_close(future, labels[:, 32:64])
        one_hot = torch.nn.functional.one_hot(labels % 4, 4)
        current_hot, future_hot = slice_subclip_targets(one_hot)
        torch.testing.assert_close(current_hot, one_hot[:, 32:33])
        torch.testing.assert_close(future_hot, one_hot[:, 32:64])
        for history_len, prediction_len in [(2, 5), (5, 2)]:
            with self.subTest(history=history_len, prediction=prediction_len):
                model = small_model(frame_memory_len=history_len, prediction_len=prediction_len)
                output = model.forward_window(torch.randn(2, history_len + 1, 12))
                self.assertEqual(output['current_logits'].shape, (2, 1, 4))
                self.assertEqual(output['future_logits'].shape, (2, prediction_len, 4))
                current, future = slice_subclip_targets(labels, history_len, prediction_len)
                self.assertEqual(current[0, 0].item(), history_len)
                torch.testing.assert_close(future, labels[:, history_len:history_len + prediction_len])

    def test_future_visual_slots_do_not_enter_training_or_evaluation(self):
        base_model = small_model(frame_memory_len=32, prediction_len=32)
        features = torch.randn(2, 64, 12)
        changed = features.clone()
        changed[:, 33:] = torch.randn_like(changed[:, 33:]) * 1000
        labels = torch.randint(0, 4, (2, 64))
        current, future = slice_subclip_targets(labels)
        batches = [dict(features=item, current_targets=current, future_targets=future) for item in [features, changed]]
        eval_losses = [evaluate_one_epoch(base_model, [batch], torch.device('cpu')) for batch in batches]
        self.assertEqual(eval_losses[0], eval_losses[1])
        trained = []
        losses = []
        for batch in batches:
            model = copy.deepcopy(base_model)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            torch.manual_seed(19)
            losses.append(train_one_epoch(model, [batch], optimizer, torch.device('cpu')))
            trained.append(model)
        self.assertEqual(losses[0], losses[1])
        for original, perturbed in zip(trained[0].parameters(), trained[1].parameters()):
            torch.testing.assert_close(original, perturbed, rtol=0, atol=0)

    def test_old_sequence_api_and_checkpoint_are_rejected(self):
        model = small_model()
        with self.assertRaisesRegex(ValueError, 'one current frame'):
            model(torch.randn(1, 64, 12))
        with self.assertRaisesRegex(ValueError, 'history frames plus one current'):
            model.forward_window(torch.randn(1, 64, 12))
        with self.assertRaises(TypeError):
            AGDSN(clip_context_len=64)
        with self.assertRaises(RuntimeError):
            model.load_state_dict({'echo_compressor.weight': torch.randn(12, 24)})


if __name__ == '__main__':
    unittest.main()
