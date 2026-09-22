from jobscan.adapters.base import SourceAdapter
from jobscan.adapters.ashby import AshbyAdapter
from jobscan.adapters.greenhouse import GreenhouseAdapter
from jobscan.adapters.lever import LeverAdapter
from jobscan.adapters.workday import WorkdayAdapter
from jobscan.models import AtsType

ADAPTERS: dict[AtsType, type[SourceAdapter]] = {
    AtsType.GREENHOUSE: GreenhouseAdapter,
    AtsType.ASHBY: AshbyAdapter,
    AtsType.LEVER: LeverAdapter,
    AtsType.WORKDAY: WorkdayAdapter,
}

__all__ = ["SourceAdapter", "GreenhouseAdapter", "AshbyAdapter", "LeverAdapter", "WorkdayAdapter", "ADAPTERS"]
