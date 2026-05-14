"""
Unit tests for MCP Weather Server (MCP-2.ipynb)
Tests cover: make_nws_request, format_alert, get_alerts, get_forecast
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import httpx

# ── Re-implement the functions under test ─────────────────────────────────────
# (Mirrors the notebook exactly so tests run standalone without Jupyter)

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get("event", "Unknown")}
Area: {props.get("areaDesc", "Unknown")}
Severity: {props.get("severity", "Unknown")}
Description: {props.get("description", "No description available")}
Instructions: {props.get("instruction", "No specific instructions provided")}
"""


async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state."""
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location."""
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:
        forecast = f"""
{period["name"]}:
Temperature: {period["temperature"]}°{period["temperatureUnit"]}
Wind: {period["windSpeed"]} {period["windDirection"]}
Forecast: {period["detailedForecast"]}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_alert_feature(
    event="Tornado Warning",
    area="Southeast Michigan",
    severity="Extreme",
    description="A tornado warning is in effect.",
    instruction="Take shelter immediately.",
) -> dict:
    return {
        "properties": {
            "event": event,
            "areaDesc": area,
            "severity": severity,
            "description": description,
            "instruction": instruction,
        }
    }


def make_forecast_period(
    name="Tonight",
    temperature=55,
    temperature_unit="F",
    wind_speed="10 mph",
    wind_direction="NW",
    detailed="Clear skies overnight.",
) -> dict:
    return {
        "name": name,
        "temperature": temperature,
        "temperatureUnit": temperature_unit,
        "windSpeed": wind_speed,
        "windDirection": wind_direction,
        "detailedForecast": detailed,
    }


def make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = status_code
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# 1. make_nws_request
# ══════════════════════════════════════════════════════════════════════════════

class TestMakeNwsRequest:

    @pytest.mark.asyncio
    async def test_successful_request_returns_json(self):
        payload = {"status": "ok", "features": []}
        mock_resp = make_mock_response(payload)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await make_nws_request("https://api.weather.gov/alerts/active/area/MI")

        assert result == payload

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self):
        mock_resp = make_mock_response({}, status_code=500)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await make_nws_request("https://api.weather.gov/bad-endpoint")

        assert result is None

    @pytest.mark.asyncio
    async def test_network_timeout_returns_none(self):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await make_nws_request("https://api.weather.gov/points/0,0")

        assert result is None

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await make_nws_request("https://api.weather.gov/alerts/active/area/CA")

        assert result is None

    @pytest.mark.asyncio
    async def test_correct_headers_sent(self):
        payload = {"features": []}
        mock_resp = make_mock_response(payload)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await make_nws_request("https://api.weather.gov/test")
            call_kwargs = mock_client.get.call_args

        headers = call_kwargs.kwargs["headers"]
        assert headers["User-Agent"] == "weather-app/1.0"
        assert headers["Accept"] == "application/geo+json"

    @pytest.mark.asyncio
    async def test_404_returns_none(self):
        mock_resp = make_mock_response({}, status_code=404)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await make_nws_request("https://api.weather.gov/nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_url_still_attempts_request(self):
        """Edge case: empty string URL should not crash — exception → None."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("invalid url"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await make_nws_request("")

        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# 2. format_alert
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatAlert:

    def test_all_fields_present(self):
        feature = make_alert_feature()
        result = format_alert(feature)
        assert "Tornado Warning" in result
        assert "Southeast Michigan" in result
        assert "Extreme" in result
        assert "A tornado warning is in effect." in result
        assert "Take shelter immediately." in result

    def test_missing_event_shows_unknown(self):
        feature = {"properties": {"areaDesc": "Ohio"}}
        result = format_alert(feature)
        assert "Event: Unknown" in result

    def test_missing_area_shows_unknown(self):
        feature = {"properties": {"event": "Flood Warning"}}
        result = format_alert(feature)
        assert "Area: Unknown" in result

    def test_missing_severity_shows_unknown(self):
        feature = {"properties": {}}
        result = format_alert(feature)
        assert "Severity: Unknown" in result

    def test_missing_description_shows_default(self):
        feature = {"properties": {}}
        result = format_alert(feature)
        assert "No description available" in result

    def test_missing_instruction_shows_default(self):
        feature = {"properties": {}}
        result = format_alert(feature)
        assert "No specific instructions provided" in result

    def test_all_fields_missing_returns_defaults(self):
        feature = {"properties": {}}
        result = format_alert(feature)
        assert "Unknown" in result
        assert "No description available" in result
        assert "No specific instructions provided" in result

    def test_empty_string_values(self):
        """Edge case: properties exist but are empty strings."""
        feature = {
            "properties": {
                "event": "",
                "areaDesc": "",
                "severity": "",
                "description": "",
                "instruction": "",
            }
        }
        result = format_alert(feature)
        # Empty strings should be used as-is (not replaced with defaults)
        assert "Event: \n" in result

    def test_none_values_handled(self):
        """Edge case: property values are explicitly None."""
        feature = {
            "properties": {
                "event": None,
                "areaDesc": None,
            }
        }
        result = format_alert(feature)
        assert "Event: None" in result

    def test_special_characters_in_fields(self):
        feature = make_alert_feature(
            description="High winds >60 mph & possible hail ≥1\"",
            instruction="Do NOT go outside. Call 911 if injured.",
        )
        result = format_alert(feature)
        assert ">" in result
        assert "&" in result

    def test_missing_properties_key_raises(self):
        """format_alert should raise KeyError if 'properties' key absent."""
        with pytest.raises(KeyError):
            format_alert({})


# ══════════════════════════════════════════════════════════════════════════════
# 3. get_alerts
# ══════════════════════════════════════════════════════════════════════════════

class TestGetAlerts:

    @pytest.mark.asyncio
    async def test_single_alert_returned(self):
        data = {"features": [make_alert_feature()]}
        with patch(f"{__name__}.make_nws_request", new=AsyncMock(return_value=data)):
            result = await get_alerts("MI")
        assert "Tornado Warning" in result
        assert "Southeast Michigan" in result

    @pytest.mark.asyncio
    async def test_multiple_alerts_joined_by_separator(self):
        data = {
            "features": [
                make_alert_feature(event="Tornado Warning"),
                make_alert_feature(event="Flood Watch"),
            ]
        }
        with patch(f"{__name__}.make_nws_request", new=AsyncMock(return_value=data)):
            result = await get_alerts("MI")
        assert "---" in result
        assert "Tornado Warning" in result
        assert "Flood Watch" in result

    @pytest.mark.asyncio
    async def test_no_active_alerts(self):
        data = {"features": []}
        with patch(f"{__name__}.make_nws_request", new=AsyncMock(return_value=data)):
            result = await get_alerts("HI")
        assert result == "No active alerts for this state."

    @pytest.mark.asyncio
    async def test_api_returns_none(self):
        with patch(f"{__name__}.make_nws_request", new=AsyncMock(return_value=None)):
            result = await get_alerts("TX")
        assert result == "Unable to fetch alerts or no alerts found."

    @pytest.mark.asyncio
    async def test_missing_features_key(self):
        """Edge case: API returns valid JSON but no 'features' key."""
        with patch(f"{__name__}.make_nws_request", new=AsyncMock(return_value={"status": "ok"})):
            result = await get_alerts("NY")
        assert result == "Unable to fetch alerts or no alerts found."

    @pytest.mark.asyncio
    async def test_correct_url_built_for_state(self):
        data = {"features": []}
        mock_req = AsyncMock(return_value=data)
        with patch(f"{__name__}.make_nws_request", new=mock_req):
            await get_alerts("CA")
        mock_req.assert_called_once_with("https://api.weather.gov/alerts/active/area/CA")

    @pytest.mark.asyncio
    async def test_lowercase_state_code(self):
        """Edge case: lowercase state code is passed through as-is."""
        data = {"features": []}
        mock_req = AsyncMock(return_value=data)
        with patch(f"{__name__}.make_nws_request", new=mock_req):
            await get_alerts("ca")
        mock_req.assert_called_once_with("https://api.weather.gov/alerts/active/area/ca")

    @pytest.mark.asyncio
    async def test_empty_state_code(self):
        """Edge case: empty string as state code."""
        data = {"features": []}
        mock_req = AsyncMock(return_value=data)
        with patch(f"{__name__}.make_nws_request", new=mock_req):
            result = await get_alerts("")
        assert result == "No active alerts for this state."

    @pytest.mark.asyncio
    async def test_invalid_state_code_api_error(self):
        """Edge case: NWS returns None for bogus state like 'ZZ'."""
        with patch(f"{__name__}.make_nws_request", new=AsyncMock(return_value=None)):
            result = await get_alerts("ZZ")
        assert result == "Unable to fetch alerts or no alerts found."

    @pytest.mark.asyncio
    async def test_features_is_none(self):
        """Edge case: features key exists but value is None.
        Because `not None` is True, the falsy-check short-circuits and returns
        the 'no active alerts' message rather than iterating and raising TypeError."""
        data = {"features": None}
        with patch(f"{__name__}.make_nws_request", new=AsyncMock(return_value=data)):
            result = await get_alerts("WA")
        assert result == "No active alerts for this state."


# ══════════════════════════════════════════════════════════════════════════════
# 4. get_forecast
# ══════════════════════════════════════════════════════════════════════════════

POINTS_RESPONSE = {
    "properties": {
        "forecast": "https://api.weather.gov/gridpoints/DTX/65,33/forecast"
    }
}


def make_forecast_response(n: int = 7) -> dict:
    """Build a forecast API response with n periods."""
    return {
        "properties": {
            "periods": [make_forecast_period(name=f"Period {i}") for i in range(n)]
        }
    }


class TestGetForecast:

    @pytest.mark.asyncio
    async def test_successful_forecast_returns_up_to_5_periods(self):
        forecast = make_forecast_response(7)

        async def mock_req(url):
            if "points" in url:
                return POINTS_RESPONSE
            return forecast

        with patch(f"{__name__}.make_nws_request", new=AsyncMock(side_effect=mock_req)):
            result = await get_forecast(42.33, -83.05)

        # 5 periods joined by "---" means 4 separators
        assert result.count("---") == 4

    @pytest.mark.asyncio
    async def test_forecast_contains_temperature_and_wind(self):
        forecast = make_forecast_response(7)

        async def mock_req(url):
            return POINTS_RESPONSE if "points" in url else forecast

        with patch(f"{__name__}.make_nws_request", new=AsyncMock(side_effect=mock_req)):
            result = await get_forecast(42.33, -83.05)

        assert "Temperature:" in result
        assert "Wind:" in result
        assert "Forecast:" in result

    @pytest.mark.asyncio
    async def test_points_api_failure_returns_error(self):
        with patch(f"{__name__}.make_nws_request", new=AsyncMock(return_value=None)):
            result = await get_forecast(42.33, -83.05)
        assert result == "Unable to fetch forecast data for this location."

    @pytest.mark.asyncio
    async def test_forecast_api_failure_returns_error(self):
        async def mock_req(url):
            return POINTS_RESPONSE if "points" in url else None

        with patch(f"{__name__}.make_nws_request", new=AsyncMock(side_effect=mock_req)):
            result = await get_forecast(42.33, -83.05)

        assert result == "Unable to fetch detailed forecast."

    @pytest.mark.asyncio
    async def test_correct_points_url_built(self):
        calls = []
        forecast = make_forecast_response(7)

        async def mock_req(url):
            calls.append(url)
            if "points" in url:
                return POINTS_RESPONSE
            return forecast

        with patch(f"{__name__}.make_nws_request", new=AsyncMock(side_effect=mock_req)):
            await get_forecast(42.33, -83.05)

        assert calls[0] == "https://api.weather.gov/points/42.33,-83.05"

    @pytest.mark.asyncio
    async def test_forecast_url_taken_from_points_response(self):
        """Ensures the second call uses the URL from points_data, not a hardcoded one."""
        calls = []
        forecast = make_forecast_response(7)

        async def mock_req(url):
            calls.append(url)
            if "points" in url:
                return POINTS_RESPONSE
            return forecast

        with patch(f"{__name__}.make_nws_request", new=AsyncMock(side_effect=mock_req)):
            await get_forecast(42.33, -83.05)

        assert calls[1] == POINTS_RESPONSE["properties"]["forecast"]

    @pytest.mark.asyncio
    async def test_fewer_than_5_periods_returned(self):
        """Edge case: API returns only 2 forecast periods."""
        short_forecast = {
            "properties": {
                "periods": [make_forecast_period(name=f"Period {i}") for i in range(2)]
            }
        }

        async def mock_req(url):
            return POINTS_RESPONSE if "points" in url else short_forecast

        with patch(f"{__name__}.make_nws_request", new=AsyncMock(side_effect=mock_req)):
            result = await get_forecast(42.33, -83.05)

        assert result.count("---") == 1  # Only 1 separator for 2 periods

    @pytest.mark.asyncio
    async def test_zero_periods_returns_empty_string(self):
        """Edge case: periods list is empty — joined result is empty string."""
        empty_forecast = {"properties": {"periods": []}}

        async def mock_req(url):
            return POINTS_RESPONSE if "points" in url else empty_forecast

        with patch(f"{__name__}.make_nws_request", new=AsyncMock(side_effect=mock_req)):
            result = await get_forecast(42.33, -83.05)

        assert result == ""

    @pytest.mark.asyncio
    async def test_boundary_coordinates(self):
        """Edge case: extreme lat/lon values (poles, date line)."""
        calls = []
        forecast = make_forecast_response(7)

        async def mock_req(url):
            calls.append(url)
            if "points" in url:
                return POINTS_RESPONSE
            return forecast

        with patch(f"{__name__}.make_nws_request", new=AsyncMock(side_effect=mock_req)):
            await get_forecast(90.0, -180.0)

        assert "90.0,-180.0" in calls[0]

    @pytest.mark.asyncio
    async def test_negative_coordinates(self):
        """Edge case: southern hemisphere / west longitude."""
        calls = []
        forecast = make_forecast_response(7)

        async def mock_req(url):
            calls.append(url)
            if "points" in url:
                return POINTS_RESPONSE
            return forecast

        with patch(f"{__name__}.make_nws_request", new=AsyncMock(side_effect=mock_req)):
            await get_forecast(-33.87, 151.21)  # Sydney, AU (NWS won't cover it)

        assert "-33.87,151.21" in calls[0]

    @pytest.mark.asyncio
    async def test_missing_forecast_key_in_points_raises(self):
        """Edge case: points response has no 'forecast' URL → KeyError."""
        bad_points = {"properties": {}}  # no "forecast" key
        forecast = make_forecast_response(7)

        async def mock_req(url):
            return bad_points if "points" in url else forecast

        with patch(f"{__name__}.make_nws_request", new=AsyncMock(side_effect=mock_req)):
            with pytest.raises(KeyError):
                await get_forecast(42.33, -83.05)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Integration-style: get_alerts + format_alert pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    @pytest.mark.asyncio
    async def test_multiple_alerts_all_formatted(self):
        features = [
            make_alert_feature(event="Tornado Warning", area="Wayne County"),
            make_alert_feature(event="Flash Flood Watch", area="Oakland County"),
            make_alert_feature(event="Winter Storm Advisory", area="Macomb County"),
        ]
        data = {"features": features}
        with patch(f"{__name__}.make_nws_request", new=AsyncMock(return_value=data)):
            result = await get_alerts("MI")

        assert "Tornado Warning" in result
        assert "Flash Flood Watch" in result
        assert "Winter Storm Advisory" in result
        assert result.count("---") == 2  # 3 alerts → 2 separators

    @pytest.mark.asyncio
    async def test_forecast_only_first_5_of_many_periods(self):
        many_periods = make_forecast_response(10)
        # Override detailed forecasts so we can check by name
        for i, p in enumerate(many_periods["properties"]["periods"]):
            p["detailedForecast"] = f"Detail {i}"

        async def mock_req(url):
            return POINTS_RESPONSE if "points" in url else many_periods

        with patch(f"{__name__}.make_nws_request", new=AsyncMock(side_effect=mock_req)):
            result = await get_forecast(42.33, -83.05)

        assert "Detail 4" in result
        assert "Detail 5" not in result  # 6th period should be excluded


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])