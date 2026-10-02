from jobscan.adapters.base import SourceAdapter
from jobscan.adapters.ashby import AshbyAdapter
from jobscan.adapters.avature import AvatureAdapter
from jobscan.adapters.esri import EsriAdapter
from jobscan.adapters.greenhouse import GreenhouseAdapter
from jobscan.adapters.jazzhr import JazzHRAdapter
from jobscan.adapters.jobvite import JobviteAdapter
from jobscan.adapters.lever import LeverAdapter
from jobscan.adapters.rippling import RipplingAdapter
from jobscan.adapters.smartrecruiters import SmartRecruitersAdapter
from jobscan.adapters.workday import WorkdayAdapter
from jobscan.models import AtsType

ADAPTERS: dict[AtsType, type[SourceAdapter]] = {
    AtsType.GREENHOUSE: GreenhouseAdapter,
    AtsType.ASHBY: AshbyAdapter,
    AtsType.LEVER: LeverAdapter,
    AtsType.WORKDAY: WorkdayAdapter,
    AtsType.JOBVITE: JobviteAdapter,
    AtsType.ESRI: EsriAdapter,
    AtsType.SMARTRECRUITERS: SmartRecruitersAdapter,
    AtsType.AVATURE: AvatureAdapter,
    AtsType.RIPPLING: RipplingAdapter,
    AtsType.JAZZHR: JazzHRAdapter,
}

__all__ = [
    "SourceAdapter", "GreenhouseAdapter", "AshbyAdapter", "LeverAdapter", "WorkdayAdapter",
    "JobviteAdapter", "EsriAdapter", "SmartRecruitersAdapter", "AvatureAdapter", "RipplingAdapter",
    "JazzHRAdapter", "ADAPTERS",
]
