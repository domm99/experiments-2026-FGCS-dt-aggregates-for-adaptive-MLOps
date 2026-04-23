import pandas as pd
from src.distributed.Simulator import Event, Monitor, Simulator


class ActivationPatientsMonitor(Monitor):

    def __init__(
        self,
        simulator: Simulator,
        activation_threshold: int,
        train_priority: int = 1,
    ) -> None:
        super().__init__(simulator)
        self._activation_threshold = activation_threshold
        self._train_priority = train_priority
        self._activations_since_last_training = 0
        self._training_pending = False

    def on_event(self, event: Event) -> None:
        if event.event_type == 'PATIENT_BECOMES_ACTIVE':
            self._activations_since_last_training += 1
            if self._training_pending:
                return
            if self._activations_since_last_training < self._activation_threshold:
                return
            self._training_pending = True
            self._simulator.schedule_event(
                Event(
                    time=event.time,
                    priority=self._train_priority,
                    event_type='TRAIN',
                    payload={'reason': 'activation_threshold'},
                )
            )
            return

        if event.event_type == 'TRAIN':
            self._activations_since_last_training = 0
            self._training_pending = False


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
        self._min_comparable_dts = min_comparable_dts
        self._threshold_mode = threshold_mode
        self._higher_is_worse = higher_is_worse
        self._train_priority = train_priority
        self._inference_priority = inference_priority
        self._baseline_metrics: dict[str, float] = {}
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
            self._training_pending = False
            self._schedule_inference(
                event.time + pd.DateOffset(days=self._inference_interval_days),
                event.time,
            )
            return

        if event.event_type != 'INFERENCE':
            return

        last_training_time = event.payload['last_training_time']
        if self._simulator.state.last_training_time != last_training_time:
            return

        comparable_results = []
        for result in self._simulator.state.last_inference_results:
            if result.get('status') != 'evaluated':
                continue
            metric_value = result.get(self._metric_name)
            if metric_value is None or pd.isna(metric_value):
                continue
            dt_id = result['dt_id']
            if dt_id not in self._baseline_metrics:
                self._baseline_metrics[dt_id] = float(metric_value)
                continue
            comparable_results.append(result)

        degraded_count = sum(
            1 for result in comparable_results
            if self._is_degraded(result['dt_id'], float(result[self._metric_name]))
        )
        comparable_count = len(comparable_results)

        if comparable_count >= self._min_comparable_dts:
            degraded_fraction = degraded_count / comparable_count
            if degraded_fraction >= self._degraded_dt_fraction_threshold and not self._training_pending:
                self._training_pending = True
                self._schedule_train(
                    event.time + pd.DateOffset(days=self._retraining_delay_days),
                    reason='performance_drift',
                )

        self._schedule_inference(
            event.time + pd.DateOffset(days=self._inference_interval_days),
            last_training_time,
        )

    def _schedule_train(self, time: pd.Timestamp, reason: str) -> None:
        self._simulator.schedule_event(
            Event(
                time=time,
                priority=self._train_priority,
                event_type='TRAIN',
                payload={'reason': reason},
            )
        )

    def _schedule_inference(self, time: pd.Timestamp, last_training_time: pd.Timestamp) -> None:
        self._simulator.schedule_event(
            Event(
                time=time,
                priority=self._inference_priority,
                event_type='INFERENCE',
                payload={'last_training_time': last_training_time},
            )
        )

    def _is_degraded(self, dt_id: str, current_metric: float) -> bool:
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


class ActivationPatientsMonitor(Monitor):

    def __init__(self, simulator: Simulator, activation_threshold: int):
        super().__init__(simulator)
        self._activated_dts = 0
        self._last_active_dts = 0
        self._activation_threshold = activation_threshold
        self._is_first_train = True
        self._last_training_time = None

    def update(self):
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
            if not self._is_first_train:
                print('========= SCHEDULING INFERENCE =========')
                test_event = Event(
                    time=current_time,
                    priority=2,
                    event_type='INFERENCE',
                    payload={'last_training_time': self._last_training_time},
                )
                self._simulator.schedule_event(test_event)

            self._last_training_time = current_time + pd.DateOffset(days=1)
            self._activated_dts = 0
            self._is_first_train = False