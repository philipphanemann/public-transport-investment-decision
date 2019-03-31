import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


opt = SolverFactory('cbc')

m = pyo.AbstractModel()
m.sizes = pyo.Set(doc='bus sizes')
m.days = pyo.Set(doc='days per week/operating days')
m.routes = pyo.Set(doc='routes')
m.Cost_fix = pyo.Param(m.sizes, doc='fixed cost per bus')
m.Seats = pyo.Param(m.sizes, doc='seats per bus size')
m.Cost_var = pyo.Param(m.sizes, doc='variable cost per route and day')
m.Demand = pyo.Param(m.days, m.routes, doc='seats to be satisfied')
m.busses = pyo.Var(m.sizes, within=pyo.NonNegativeIntegers, doc='total busses per size')
m.busses_disp = pyo.Var(m.sizes, m.routes, m.days,
                        within=pyo.NonNegativeIntegers,
                        doc='dispatched busses')


data = pyo.DataPortal()
data.load(filename='busses.csv', index=m.sizes,
          param=(m.Seats, m.Cost_fix, m.Cost_var))
dem = pd.read_csv('demand.csv', index_col=0)
data.data().update({'days': {None: dem.index.tolist()}})
data.data().update({'routes': {None: dem.columns.tolist()}})
data.data().update({'Demand': dem.unstack().swaplevel().to_dict()})


def obj_expression(m):
    return (sum(m.Cost_fix[size] * m.busses[size] for size in m.sizes)
            +
            sum(m.busses_disp[size, route, day] * m.Cost_var[size]
                  for size in m.sizes
                  for route in m.routes
                  for day in m.days))


def demand_satisfaction(m, day, route):
    return m.Demand[day, route] <= sum(m.Seats[size] * m.busses_disp[size, route, day]
                                       for size in m.sizes)


def bus_dispatch(m, size, day):
    return sum(m.busses_disp[size, r, day] for r in m.routes) <= m.busses[size]


m.obj = pyo.Objective(rule=obj_expression, sense=pyo.minimize)
m.dem_con = pyo.Constraint(m.days, m.routes, rule=demand_satisfaction)
m.bus_con = pyo.Constraint(m.sizes, m.days, rule=bus_dispatch)


instance = m.create_instance(data)

results = opt.solve(instance, symbolic_solver_labels=True, tee=True,
                    load_solutions=True)

