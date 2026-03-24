from .paithon_runtime import engine


@engine.response_adapter
def parse_tax_rate(payload: dict) -> float:
    """Return a tax rate as a float from an external tax API payload.

    Tolerate reasonable response-shape drift across nested dict and object-like variants, but fail for missing tax data.
    """
    return float(payload["tax"]["rate"]["value"])
