"""probe_board: a candidate board run through the real adapter, title gate and factual filters,
with nothing persisted — what `companies probe` prints before a company is added."""

from __future__ import annotations

import httpx
import pytest
import respx

from jobscan.adapters.base import AdapterError
from jobscan.cli import main
from jobscan.models import AtsType
from jobscan.probe import probe_board

BOARD = "https://boards-api.greenhouse.io/v1/boards/acme/jobs"


def job(job_id, title, location, content):
    return {
        "id": job_id,
        "title": title,
        "location": {"name": location},
        "absolute_url": f"https://acme.example.com/jobs/{job_id}",
        "content": content,
        "metadata": [],
        "departments": [],
    }


@respx.mock
def test_probe_reports_what_would_reach_evaluation(settings):
    respx.get(BOARD).mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    job(1, "Account Executive", "Remote - US", "<p>$150,000 - $200,000</p>"),
                    job(
                        2,
                        "Senior Backend Engineer",
                        "Hybrid - San Francisco, CA",
                        "<p>Three days a week in office. $190,000 - $230,000</p>",
                    ),
                    job(
                        3,
                        "Senior Software Engineer",
                        "Remote - US",
                        "<p>Python and SQL. $180,000 - $220,000</p>",
                    ),
                ]
            },
        )
    )
    with httpx.Client() as client:
        result = probe_board(settings, AtsType.GREENHOUSE, "acme", client)

    assert (result.postings, result.engineering) == (3, 2)
    assert [j.title for j in result.passing] == ["Senior Software Engineer"]
    assert result.rejected == {"not_remote_explicit": 1}


@respx.mock
def test_probe_surfaces_a_board_that_cannot_be_fetched(settings):
    respx.get(BOARD).mock(return_value=httpx.Response(404))
    with httpx.Client() as client, pytest.raises(AdapterError):
        probe_board(settings, AtsType.GREENHOUSE, "acme", client)


def test_cli_rejects_an_ats_without_an_adapter(capsys):
    assert main(["companies", "probe", "taleo", "acme"]) == 1
    assert "No adapter for taleo" in capsys.readouterr().err
