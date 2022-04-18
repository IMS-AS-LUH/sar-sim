from time import perf_counter as _perf_counter


class TimeStamper(object):
    _TIMESPEC = [
        (1e-9, 'ns', 1e4),
        (1e-6, 'us', 1e4),
        (1e-3, 'ms', 1e4),
        (1, 'sec', 600),
        (60, 'min', 240),
        (60 * 60, 'hrs', None)
    ]

    def __init__(self, report_tic_inline: bool = True, report_toc_inline: bool = True):
        self._timestamps = []
        self._tic = None
        self._section = None
        self.report_toc_inline = report_toc_inline
        self.report_tic_inline = report_tic_inline
        self._last_timestamp = None

    def tic(self, section: str):
        self.toc()
        self._section = section
        if self.report_tic_inline and section is not None:
            print(f'⏱ Start of {section}')
        self._tic = _perf_counter()

    def toc(self):
        _toc = _perf_counter()
        if self._tic is not None:
            self._last_timestamp = dict(
                section=self._section,
                tic=self.tic,
                toc=_toc,
                time=_toc - self._tic
            )
            self._timestamps.append(self._last_timestamp)
            self._tic = None
            self._section = None
            if self.report_toc_inline:
                self.report_last()

    def report_last(self):
        if self._last_timestamp is not None:
            self._report(self._last_timestamp)

    def _report(self, timestamp: dict):
        _time = timestamp['time']
        _unit = 's'
        for (factor, name, limit) in self._TIMESPEC:
            if limit is None or _time < limit * factor:
                _time = _time / factor
                _unit = name
                break
        print(f'⏱ {timestamp["section"] or "This"} took {_time:.2f} {_unit}')

    def report_all(self):
        print('*** Timestamp Summary ***')
        for timestamp in self._timestamps:
            self._report(timestamp)
        print('*************************')

    def get_timestamps(self) -> list:
        return self._timestamps

    def clear(self) -> None:
        self._timestamps = []
        self._last_timestamp = None
        self._tic = None
