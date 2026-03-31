from dataclasses import dataclass
from typing import ClassVar
import porepy as pp


@dataclass(kw_only=True, eq=False)
class DamageSolid(pp.SolidConstants):
    SI_units: ClassVar[dict[str, str]] = pp.SolidConstants.SI_units
    SI_units.update(
        {
            "initial_friction_damage": "-",
            "initial_dilation_damage": "-",
            "characteristic_fracture_roughness": "m",
            "universal_compressive_strength": "Pa",
        }
    )
    initial_friction_damage: 1.0
    initial_dilation_damage: 1.0
    characteristic_fracture_roughness: 1.0
    universal_compressive_strength: 1.0
