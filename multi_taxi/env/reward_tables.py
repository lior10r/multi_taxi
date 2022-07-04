from ..utils.types import Event

TAXI_ENVIRONMENT_REWARDS = {
    Event.STEP: -1,
    Event.MOVE: 0,
    Event.USE_ENGINE_WHILE_NO_FUEL: 0,
    Event.BAD_PICKUP: 0,
    Event.BAD_DROPOFF: 0,
    Event.BAD_REFUEL: 0,
    Event.BAD_FUEL: 0,
    Event.PICKUP: 0,
    Event.TURN_ENGINE_ON: 0,
    Event.TURN_ENGINE_OFF: 0,
    Event.STANDBY_ENGINE_ON: 0,
    Event.STANDBY_ENGINE_OFF: 0,
    Event.INTERMEDIATE_DROPOFF: 0,
    Event.FINAL_DROPOFF: 100,
    Event.HIT_OBSTACLE: 0,
    Event.COLLISION: -100,
    Event.STUCK_WITHOUT_FUEL: -100,
    Event.OUT_OF_TIME: 0,
    Event.DEAD: -1,
    Event.USE_ENGINE_WHILE_OFF: 0,
    Event.ENGINE_ALREADY_ON: 0,
    Event.REFUEL: 0
}

PICKUP_ONLY_TAXI_ENVIRONMENT_REWARDS = {
    Event.STEP: -1,
    Event.MOVE: 0,
    Event.USE_ENGINE_WHILE_NO_FUEL: 0,
    Event.BAD_PICKUP: 0,
    Event.BAD_REFUEL: 0,
    Event.BAD_FUEL: 0,
    Event.PICKUP: 100,
    Event.TURN_ENGINE_ON: 0,
    Event.TURN_ENGINE_OFF: 0,
    Event.STANDBY_ENGINE_ON: 0,
    Event.STANDBY_ENGINE_OFF: 0,
    Event.HIT_OBSTACLE: 0,
    Event.COLLISION: -100,
    Event.STUCK_WITHOUT_FUEL: -100,
    Event.OUT_OF_TIME: 0,
    Event.DEAD: -1,
    Event.USE_ENGINE_WHILE_OFF: 0,
    Event.ENGINE_ALREADY_ON: 0,
    Event.REFUEL: 0
}
