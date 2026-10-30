from jobscan.adapters.ashby import AshbyAdapter
from jobscan.adapters.avature import AvatureAdapter
from jobscan.adapters.base import SourceAdapter
from jobscan.adapters.directemployers import DirectEmployersAdapter
from jobscan.adapters.esri import EsriAdapter
from jobscan.adapters.greenhouse import GreenhouseAdapter
from jobscan.adapters.icims import ICIMSAdapter
from jobscan.adapters.jazzhr import JazzHRAdapter
from jobscan.adapters.jibe import JibeAdapter
from jobscan.adapters.jobvite import JobviteAdapter
from jobscan.adapters.lever import LeverAdapter
from jobscan.adapters.oracle import OracleAdapter
from jobscan.adapters.pinpoint import PinpointAdapter
from jobscan.adapters.rippling import RipplingAdapter
from jobscan.adapters.smartrecruiters import SmartRecruitersAdapter
from jobscan.adapters.workable import WorkableAdapter
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
    AtsType.ICIMS: ICIMSAdapter,
    AtsType.WORKABLE: WorkableAdapter,
    AtsType.JIBE: JibeAdapter,
    AtsType.ORACLE: OracleAdapter,
    AtsType.DIRECTEMPLOYERS: DirectEmployersAdapter,
    AtsType.PINPOINT: PinpointAdapter,
}

__all__ = [
    "ADAPTERS",
    "AshbyAdapter",
    "AvatureAdapter",
    "DirectEmployersAdapter",
    "EsriAdapter",
    "GreenhouseAdapter",
    "ICIMSAdapter",
    "JazzHRAdapter",
    "JibeAdapter",
    "JobviteAdapter",
    "LeverAdapter",
    "OracleAdapter",
    "PinpointAdapter",
    "RipplingAdapter",
    "SmartRecruitersAdapter",
    "SourceAdapter",
    "WorkableAdapter",
    "WorkdayAdapter",
]
