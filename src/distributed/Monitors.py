from pathlib import Path

import pandas as pd
from src.distributed.Simulator import Event, Monitor, Simulator


# class ActivationPatientsMonitor(Monitor):
#
#     def __init__(
#         self,
#         simulator: Simulator,
#         activation_threshold: int,
#         train_priority: int = 1,
#     ) -> None:
#         super().__init__(simulator)
#         self._activation_threshold = activation_threshold
#         self._train_priority = train_priority
#         self._activations_since_last_training = 0
#         self._training_pending = False
#
#     def on_event(self, event: Event) -> None:
#         if event.event_type == 'PATIENT_BECOMES_ACTIVE':
#             self._activations_since_last_training += 1
#             if self._training_pending:
#                 return
#             if self._activations_since_last_training < self._activation_threshold:
#                 return
#             self._training_pending = True
#             self._simulator.schedule_event(
#                 Event(
#                     time=event.time,
#                     priority=self._train_priority,
#                     event_type='TRAIN',
#                     payload={'reason': 'activation_threshold'},
#                 )
#             )
#             return
#
#         if event.event_type == 'TRAIN':
#             self._activations_since_last_training = 0
#             self._training_pending = False


class PerformanceDriftMonitor(Monitor):

    def __init__(
        self,
        simulator: Simulator,
        bootstrap_months: int,
        inference_interval_days: int,
        retraining_delay_days: int,
        metric_name: str,
        degradation_threshold: float,
        degraded_dt_fraction_threshold: float,
        metric_floor: float | None = None,
        min_comparable_dts: int = 1,
        threshold_mode: str = 'relative',
        higher_is_worse: bool = True,
        train_priority: int = 1,
        inference_priority: int = 2,
    ) -> None:
        super().__init__(simulator)
        self._bootstrap_months = bootstrap_months
        self._inference_interval_days = inference_interval_days
        self._retraining_delay_days = retraining_delay_days
        self._metric_name = metric_name
        self._degradation_threshold = degradation_threshold
        self._degraded_dt_fraction_threshold = degraded_dt_fraction_threshold
        self._metric_floor = metric_floor
        self._min_comparable_dts = min_comparable_dts
        self._threshold_mode = threshold_mode
        self._higher_is_worse = higher_is_worse
        self._train_priority = train_priority
        self._inference_priority = inference_priority
        self._baseline_metrics: dict[str, float] = {}
        self._baseline_timestamps: dict[str, pd.Timestamp] = {}
        self._training_pending = False

    def on_start(self) -> None:
        self._schedule_train(
            self._simulator.time + pd.DateOffset(months=self._bootstrap_months),
            reason='bootstrap_window',
        )

    def on_event(self, event: Event) -> None:
        if event.event_type == 'TRAIN':
            if self._simulator.state.last_training_time != event.time:
                self._training_pending = False
                return
            self._baseline_metrics = {}
            self._baseline_timestamps = {}
            self._training_pending = False
            self._schedule_inference(
                event.time + pd.DateOffset(days=self._inference_interval_days),
                event.time,
            )
            return

        if event.event_type != 'INFERENCE':
            return

        last_training_time = event.payload['last_training_time']
        detection_time = event.time
        if self._simulator.state.last_training_time != last_training_time:
            return

        evaluated_results = []
        for result in self._simulator.state.last_inference_results:
            if result.get('status') != 'evaluated':
                continue
            metric_value = result.get(self._metric_name)
            if metric_value is None or pd.isna(metric_value):
                continue
            evaluated_results.append(result)

        if self._metric_floor is not None:
            comparable_results = evaluated_results
        else:
            comparable_results = []
            for result in evaluated_results:
                dt_id = result['dt_id']
                metric_value = float(result[self._metric_name])
                if dt_id not in self._baseline_metrics:
                    self._baseline_metrics[dt_id] = metric_value
                    self._baseline_timestamps[dt_id] = detection_time
                    continue
                comparable_results.append(result)

        degraded_results = [
            result for result in comparable_results
            if self._is_degraded(result['dt_id'], float(result[self._metric_name]))
        ]
        degraded_count = len(degraded_results)
        comparable_count = len(comparable_results)

        if comparable_count >= self._min_comparable_dts:
            degraded_fraction = degraded_count / comparable_count
            if degraded_fraction >= self._degraded_dt_fraction_threshold and not self._training_pending:
                reason = 'low_accuracy' if self._metric_floor is not None else 'performance_drift'
                scheduled_training_time = detection_time + pd.DateOffset(days=self._retraining_delay_days)
                train_event_scheduled = self._schedule_train(
                    scheduled_training_time,
                    reason=reason,
                )
                self._training_pending = train_event_scheduled
                self._export_drift_event(
                    last_training_time=last_training_time,
                    detection_time=detection_time,
                    scheduled_training_time=scheduled_training_time,
                    train_event_scheduled=train_event_scheduled,
                    comparable_results=comparable_results,
                    degraded_results=degraded_results,
                    degraded_fraction=degraded_fraction,
                    reason=reason,
                )

        self._schedule_inference(
            detection_time + pd.DateOffset(days=self._inference_interval_days),
            last_training_time,
        )

    def _schedule_train(self, time: pd.Timestamp, reason: str) -> bool:
        return self._simulator.schedule_event(
            Event(
                time=time,
                priority=self._train_priority,
                event_type='TRAIN',
                payload={'reason': reason},
            )
        )

    def _schedule_inference(self, time: pd.Timestamp, last_training_time: pd.Timestamp) -> bool:
        return self._simulator.schedule_event(
            Event(
                time=time,
                priority=self._inference_priority,
                event_type='INFERENCE',
                payload={'last_training_time': last_training_time},
            )
        )

    def _reference_metric(self, dt_id: str) -> float:
        if self._metric_floor is not None:
            return float(self._metric_floor)
        return self._baseline_metrics[dt_id]

    def _absolute_degradation(self, dt_id: str, current_metric: float) -> float:
        reference_metric = self._reference_metric(dt_id)
        if self._higher_is_worse:
            return current_metric - reference_metric
        return reference_metric - current_metric

    def _relative_degradation(self, dt_id: str, current_metric: float) -> float:
        reference_metric = self._reference_metric(dt_id)
        degradation = self._absolute_degradation(dt_id, current_metric)
        reference_scale = abs(reference_metric)
        if reference_scale == 0:
            return 1.0 if degradation > 0 else 0.0
        return degradation / reference_scale

    def _mean_or_nan(self, values: list[float]) -> float:
        if not values:
            return float('nan')
        return float(sum(values) / len(values))

    def _days_between(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> float:
        delta = end_time - start_time
        return float(delta.total_seconds() / 86400.0)

    def _summarize_results(
        self,
        results: list[dict],
        reference_time: pd.Timestamp,
    ) -> dict[str, float]:
        current_metrics: list[float] = []
        reference_metrics: list[float] = []
        absolute_degradations: list[float] = []
        relative_degradations: list[float] = []
        reference_ages_days: list[float] = []

        for result in results:
            dt_id = result['dt_id']
            current_metric = float(result[self._metric_name])
            current_metrics.append(current_metric)
            reference_metrics.append(self._reference_metric(dt_id))
            absolute_degradations.append(self._absolute_degradation(dt_id, current_metric))
            relative_degradations.append(self._relative_degradation(dt_id, current_metric))

            baseline_time = self._baseline_timestamps.get(dt_id)
            if baseline_time is not None:
                reference_ages_days.append(self._days_between(baseline_time, reference_time))

        return {
            'mean_current_metric': self._mean_or_nan(current_metrics),
            'mean_reference_metric': self._mean_or_nan(reference_metrics),
            'mean_absolute_degradation': self._mean_or_nan(absolute_degradations),
            'mean_relative_degradation': self._mean_or_nan(relative_degradations),
            'mean_reference_age_days': self._mean_or_nan(reference_ages_days),
            'max_reference_age_days': max(reference_ages_days) if reference_ages_days else float('nan'),
        }

    def _export_drift_event(
        self,
        last_training_time: pd.Timestamp,
        detection_time: pd.Timestamp,
        scheduled_training_time: pd.Timestamp,
        train_event_scheduled: bool,
        comparable_results: list[dict],
        degraded_results: list[dict],
        degraded_fraction: float,
        reason: str,
    ) -> None:
        comparable_summary = self._summarize_results(comparable_results, detection_time)
        degraded_summary = self._summarize_results(degraded_results, detection_time)
        metrics = {
            'reason': reason,
            'metric_name': self._metric_name,
            'threshold_mode': self._threshold_mode,
            'metric_floor': self._metric_floor,
            'degradation_threshold': self._degradation_threshold,
            'degraded_dt_fraction_threshold': self._degraded_dt_fraction_threshold,
            'higher_is_worse': self._higher_is_worse,
            'last_training_time': last_training_time,
            'detection_time': detection_time,
            'scheduled_training_time': scheduled_training_time,
            'train_event_scheduled': train_event_scheduled,
            'detection_latency_days': self._days_between(last_training_time, detection_time),
            'schedule_latency_days': self._days_between(detection_time, scheduled_training_time),
            'end_to_end_latency_days': self._days_between(last_training_time, scheduled_training_time),
            'comparable_dt_count': len(comparable_results),
            'degraded_dt_count': len(degraded_results),
            'degraded_fraction': degraded_fraction,
            'degraded_dt_ids': '|'.join(sorted(result['dt_id'] for result in degraded_results)),
            'simulation_end_time': self._simulator.ending_time,
            **comparable_summary,
            'mean_degraded_current_metric': degraded_summary['mean_current_metric'],
            'mean_degraded_reference_metric': degraded_summary['mean_reference_metric'],
            'mean_degraded_absolute_degradation': degraded_summary['mean_absolute_degradation'],
            'mean_degraded_relative_degradation': degraded_summary['mean_relative_degradation'],
            'mean_degraded_reference_age_days': degraded_summary['mean_reference_age_days'],
            'max_degraded_reference_age_days': degraded_summary['max_reference_age_days'],
        }

        output_dir = Path(self._simulator.config.data_export_path) / self._simulator.experiment
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'drift_events-seed_{self._simulator.seed}.csv'

        if output_path.exists():
            metrics_df = pd.read_csv(output_path)
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
        else:
            metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_path, index=False)

    def _is_degraded(self, dt_id: str, current_metric: float) -> bool:
        if self._metric_floor is not None:
            if self._higher_is_worse:
                return current_metric > self._metric_floor
            return current_metric < self._metric_floor

        baseline_metric = self._baseline_metrics[dt_id]
        if self._higher_is_worse:
            delta = current_metric - baseline_metric
        else:
            delta = baseline_metric - current_metric

        if self._threshold_mode == 'absolute':
            return delta >= self._degradation_threshold

        baseline_scale = abs(baseline_metric)
        if baseline_scale == 0:
            return delta > 0
        return (delta / baseline_scale) >= self._degradation_threshold


class PeriodicInferenceMonitor(Monitor):

    def __init__(
        self,
        simulator: Simulator,
        inference_interval_days: int,
        inference_priority: int = 2,
    ) -> None:
        super().__init__(simulator)
        self._inference_interval_days = inference_interval_days
        self._inference_priority = inference_priority
        self._source = 'periodic_evaluation'

    def on_event(self, event: Event) -> None:
        if event.event_type == 'TRAIN':
            if self._simulator.state.last_training_time == event.time:
                self._schedule_next_inference(event.time, event.time)
            return

        if event.event_type != 'INFERENCE':
            return

        if event.payload.get('source') != self._source:
            return

        last_training_time = event.payload['last_training_time']
        if self._simulator.state.last_training_time != last_training_time:
            return

        self._schedule_next_inference(event.time, last_training_time)

    def _schedule_next_inference(
        self,
        current_time: pd.Timestamp,
        last_training_time: pd.Timestamp,
    ) -> bool:
        return self._simulator.schedule_event(
            Event(
                time=current_time + pd.DateOffset(days=self._inference_interval_days),
                priority=self._inference_priority,
                event_type='INFERENCE',
                payload={
                    'last_training_time': last_training_time,
                    'source': self._source,
                },
            )
        )


class ActivationPatientsMonitor(Monitor):

    def __init__(self, simulator: Simulator, activation_threshold: int):
        super().__init__(simulator)
        self._activated_dts = 0
        self._last_active_dts = 0
        self._activation_threshold = activation_threshold

    def on_event(self, event: Event):
        active_dts = len(self._simulator._state.active_patients)
        delta = active_dts - self._last_active_dts
        self._last_active_dts = active_dts
        self._activated_dts += delta

        if self._activated_dts > self._activation_threshold:
            ### Enough DTs have been activated, so I'm going to schedule a train
            print('========= SCHEDULING TRAIN =========')
            current_time = self._simulator.time

            train_event = Event(
                time=current_time + pd.DateOffset(days=1),
                priority=2,
                event_type='TRAIN',
                payload={},
            )
            self._simulator.schedule_event(train_event)

            self._activated_dts = 0
