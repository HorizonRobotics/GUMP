from nuplan.common.actor_state.vehicle_parameters import VehicleParameters


def get_ideal_one_parameters() -> VehicleParameters:
    return VehicleParameters(
        vehicle_name="Leading Ideal One",
        vehicle_type="gen1",
        width=1.96,
        front_length=3.9,
        rear_length=1.13,
        wheel_base=2.935,
        cog_position_from_rear_axle=1.67,
        height=1.777,
    )
