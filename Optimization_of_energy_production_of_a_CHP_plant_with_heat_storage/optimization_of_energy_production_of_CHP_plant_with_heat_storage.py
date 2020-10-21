from gurobipy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pathlib import *

TIME_HORIZON = (
    24 * 7
)  # time horizon is seven days - one week, with time-step of one hour
MAX_HEAT_DEMAND = 60  # maximal heat demand is 80MWh, while max CHP capacity is 70MWh
MAX_CHP_POWER_CAPACITY = 35  # MWh
MAX_CHP_HEAT_CAPACITY = 70  # MWh
MAX_FUEL_CONSUMPTION = 127.27  # MWh
MAX_STORAGE_CAPACITY = [i for i in range(0, 200, 10)]  # MWh
EFF_STOR = 0.9
EFF_DIS = 0.8
FUEL_PRICE = 15  # e/MWh
KEYS = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
EXTREME_POINTS_POWER = dict(
    [(KEYS[0], 10), (KEYS[1], 10), (KEYS[2], 50), (KEYS[3], 50), (KEYS[4], 30)]
)
EXTREME_POINTS_HEAT = dict(
    [(KEYS[0], 40), (KEYS[1], 7), (KEYS[2], 4), (KEYS[3], 54), (KEYS[4], 80)]
)
EXTREME_POINTS_FUEL = dict(
    [(KEYS[0], 10), (KEYS[1], 10), (KEYS[2], 70), (KEYS[3], 90), (KEYS[4], 95)]
)
DATA_PATH = Path(__file__).parent.absolute() / "data"
heat_demand_total = pd.read_csv(
    DATA_PATH / "total_heat_demand.csv",
    usecols=["NL_heat_demand_total"],
    nrows=TIME_HORIZON,
)
electricity_price = pd.read_csv(DATA_PATH / "day_ahead_electricity_prices.csv")
heat_demand = heat_demand_total * MAX_HEAT_DEMAND / heat_demand_total.max()
objective_function_value = []
mean_squared_errors = []


def get_optimized_values(EXTREME_POINTS, x_optimized):
    extreme_points = np.array(list(EXTREME_POINTS.values()))
    optimized_values = (x_optimized * extreme_points).sum(axis=1)
    return optimized_values


def model(max_storage_capacity):
    m = Model("CHPED")
    x = m.addVars(TIME_HORIZON, len(KEYS), name="x")
    q_stor = m.addVars(TIME_HORIZON, name="q_stor")
    q_dis = m.addVars(TIME_HORIZON, name="q_dis")
    coefficient_sum_constraints = m.addConstrs(
        (x.sum(i, "*") == 1 for i in range(TIME_HORIZON)),
        name="coefficient_sum_constraint",
    )
    coefficient_lower_constraint = m.addConstrs(
        (0 <= x[i, j] for i in range(TIME_HORIZON) for j in range(len(KEYS))),
        name="coefficient_lower_constraint",
    )
    power_constraints = m.addConstrs(
        (
            tupledict(zip(KEYS, x.select(i, "*"))).prod(EXTREME_POINTS_POWER, "*", "*")
            <= MAX_CHP_POWER_CAPACITY
            for i in range(TIME_HORIZON)
        ),
        name="power_constraints",
    )
    heat_constraints = m.addConstrs(
        (
            tupledict(zip(KEYS, x.select(i, "*"))).prod(EXTREME_POINTS_HEAT, "*", "*")
            <= MAX_CHP_HEAT_CAPACITY
            for i in range(TIME_HORIZON)
        ),
        name="heat_constraints",
    )
    fuel_constraints = m.addConstrs(
        (
            tupledict(zip(KEYS, x.select(i, "*"))).prod(EXTREME_POINTS_FUEL, "*", "*")
            <= MAX_FUEL_CONSUMPTION
            for i in range(TIME_HORIZON)
        ),
        name="fuel_constraints",
    )
    heat_demand_constraints = m.addConstrs(
        (
            tupledict(zip(KEYS, x.select(i, "*"))).prod(EXTREME_POINTS_HEAT, "*", "*")
            - q_stor[i]
            + EFF_DIS * q_dis[i]
            == heat_demand.iloc[i]["NL_heat_demand_total"]
            for i in range(TIME_HORIZON)
        ),
        name="heat_demand_constraints",
    )
    q_stor_initial_constraint = m.addConstr(
        q_stor[0] == 0, name="q_stor_initial_constraint"
    )
    q_dis_initial_constraint = m.addConstr(
        q_dis[0] == 0, name="q_dis_initial_constraint"
    )
    heat_storage = [0] * (TIME_HORIZON + 1)
    for i in range(1, TIME_HORIZON):
        heat_storage[i] = EFF_STOR * heat_storage[i - 1] + q_stor[i] - q_dis[i]
    heat_storage_upper_constraint = m.addConstrs(
        (heat_storage[i] <= max_storage_capacity for i in range(TIME_HORIZON)),
        name="heat_storage_upper_constraint",
    )
    heat_storage_lower_constraint = m.addConstrs(
        (0 <= heat_storage[i] for i in range(TIME_HORIZON)),
        name="heat_storage_lower_constraint",
    )
    m.setObjective(
        (
            quicksum(
                FUEL_PRICE
                * tupledict(zip(KEYS, x.select(i, "*"))).prod(
                    EXTREME_POINTS_FUEL, "*", "*"
                )
                - electricity_price["Day_ahead_price"][i]
                * tupledict(zip(KEYS, x.select(i, "*"))).prod(
                    EXTREME_POINTS_POWER, "*", "*"
                )
                for i in range(TIME_HORIZON)
            )
        ),
        GRB.MINIMIZE,
    )
    m.write("CHPED.lp")
    m.optimize()
    obj = m.getObjective()
    objective_function_value.append(obj.getValue())

    x_optimized = []
    v = m.getVars()
    for i in range(TIME_HORIZON * len(KEYS)):
        x_optimized.append(v[i].x)
    x_optimized = np.reshape(x_optimized, (TIME_HORIZON, len(KEYS)))
    fuel_consumption_optimized = get_optimized_values(EXTREME_POINTS_FUEL, x_optimized)
    heat_demand_optimized = get_optimized_values(EXTREME_POINTS_HEAT, x_optimized)
    mean_squared_errors.append(
        mean_squared_error(
            heat_demand["NL_heat_demand_total"].values.tolist(),
            heat_demand_optimized.tolist(),
        )
    )

    power_production_optimized = get_optimized_values(EXTREME_POINTS_POWER, x_optimized)
    # visualization of results
    plt.plot(fuel_consumption_optimized)
    plt.xlabel("Time [h]")
    plt.ylabel("Fuel consumption [MWh]")
    plt.fill_between(range(0, TIME_HORIZON), fuel_consumption_optimized)
    plt.title("Fuel consumption")
    plt.show()

    plt.plot(power_production_optimized)
    plt.xlabel("Time [h]")
    plt.ylabel("Produced power [MWh]")
    plt.fill_between(range(0, TIME_HORIZON), power_production_optimized)
    plt.title("Power production")
    plt.show()

    plt.plot(heat_demand_optimized)
    plt.plot(heat_demand)
    plt.legend(["Produced heat", "Real heat demand"])
    plt.xlabel("Time [h]")
    plt.ylabel("Heat [MWh]")
    plt.show()


for i in range(len(MAX_STORAGE_CAPACITY)):
    model(MAX_STORAGE_CAPACITY[i])

plt.plot(MAX_STORAGE_CAPACITY, objective_function_value)
plt.xlabel("Heat storage capacity [MW]")
plt.ylabel("Net acquisition cost [e]")
plt.grid()
plt.show()

plt.plot(MAX_STORAGE_CAPACITY, mean_squared_errors)
plt.xlabel("Heat storage capacity [MW]")
plt.ylabel("Mean squared error of produced heat and required heat")
plt.grid()
plt.show()
