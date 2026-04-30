import numpy as np
import jax
import jax.numpy as jnp
from jax import jacobian
from functools import partial

# Enable 64-bit precision for power flow accuracy
jax.config.update("jax_enable_x64", True)


def ybus(bus_data, branch_data, baseMVA=100.0):
    """Build the bus admittance matrix using the pi-line model."""
    num_buses = max(bus['bus_i'] for bus in bus_data)
    Ybus = np.zeros((num_buses, num_buses), dtype=complex)

    for branch in branch_data:
        f, t = branch['fbus'] - 1, branch['tbus'] - 1
        z = complex(branch['r'], branch['x'])
        if abs(z) < 1e-10:
            z = 1j * 1e-6
        y_series = 1.0 / z
        y_shunt = 1j * branch['b'] / 2.0

        Ybus[f, f] += y_series + y_shunt
        Ybus[t, t] += y_series + y_shunt
        Ybus[f, t] -= y_series
        Ybus[t, f] -= y_series

    for bus in bus_data:
        i = bus['bus_i'] - 1
        Ybus[i, i] += complex(bus.get('Gs', 0.0), bus.get('Bs', 0.0)) / baseMVA

    return Ybus


def calculate_mismatch(x_flat, *, Ybus, P_spec, Q_spec, non_slack, pq, Va_init, Vm_init):
    """
    Mismatch function is written with JAX so the Jacobian matrix isobtained automatically
    by differentiation instead of coding the partial derivatives by hand.
    """
    # Reconstruct full Va and Vm vectors from the flat state vector
    va = jnp.array(Va_init).at[non_slack].set(x_flat[:len(non_slack)])
    vm = jnp.array(Vm_init).at[pq].set(x_flat[len(non_slack):])

    # Complex voltages
    V = vm * jnp.exp(1j * va)

    # Calculated power injections
    I = Ybus @ V
    S_calc = V * jnp.conj(I)

    dP = jnp.real(S_calc) - P_spec
    dQ = jnp.imag(S_calc) - Q_spec

    # Only the mismatches we actually solve for
    return jnp.concatenate([dP[non_slack], dQ[pq]])


def newton_raphson(bus_data, branch_data, baseMVA, max_iter, tol, damping):
    """Newton-Raphson power flow solver using JAX for the Jacobian matrix."""
    num_buses = max(bus['bus_i'] for bus in bus_data)
    Ybus_np = ybus(bus_data, branch_data, baseMVA)          # NumPy version for final calculations
    

    # Prepare indices and static data
    bus_types = np.array([bus.get('type', 1) for bus in bus_data])
    pv = np.where(bus_types == 2)[0]
    pq = np.where(bus_types == 1)[0]
    non_slack = np.concatenate((pv, pq))

    P_spec = np.array([(bus.get('Pg', 0.0) - bus.get('Pd', 0.0)) / baseMVA for bus in bus_data])
    Q_spec = np.array([(bus.get('Qg', 0.0) - bus.get('Qd', 0.0)) / baseMVA for bus in bus_data])

    Va_init = np.array([np.deg2rad(bus.get('Va', 0.0)) for bus in bus_data]) # immediately converts initial angles to radians for the internal solver state
    Vm_init = np.array([bus.get('Vm', 1.0) for bus in bus_data])

    # Convert everything to JAX arrays for the mismatch function
    Ybus_jnp = jnp.array(Ybus_np, dtype=jnp.complex128)
    P_spec_jnp = jnp.array(P_spec, dtype=jnp.float64)
    Q_spec_jnp = jnp.array(Q_spec, dtype=jnp.float64)
    non_slack_jnp = jnp.array(non_slack, dtype=jnp.int32)
    pq_jnp = jnp.array(pq, dtype=jnp.int32)
    Va_init_jnp = jnp.array(Va_init, dtype=jnp.float64)
    Vm_init_jnp = jnp.array(Vm_init, dtype=jnp.float64)

    # Initial flat state vector
    x = jnp.concatenate([Va_init_jnp[non_slack_jnp], Vm_init_jnp[pq_jnp]])

    # Create the JAX mismatch function
    mismatch_fn = partial(
        calculate_mismatch,
        Ybus=Ybus_jnp,
        P_spec=P_spec_jnp,
        Q_spec=Q_spec_jnp,
        non_slack=non_slack_jnp,
        pq=pq_jnp,
        Va_init=Va_init_jnp,
        Vm_init=Vm_init_jnp,
    )

    # JAX automatically builds and JIT-compiles the Jacobian
    get_jacobian = jax.jit(jacobian(mismatch_fn))

    success = False
    iterations = 0
    max_mismatch_val = 1e10

    for i in range(max_iter):
        F = mismatch_fn(x)
        max_err = float(jnp.max(jnp.abs(F)))
        print(f"Iteration {i+1}: Max Mismatch = {max_err:.2e}")

        iterations = i + 1
        max_mismatch_val = max_err

        if max_err < tol:
            print("Converged!")
            success = True
            break

        J = get_jacobian(x)
        dx = jnp.linalg.solve(J, -F)
        x = x + damping * dx

    if not success:
        print(f"Did not converge within {max_iter} iterations (final mismatch = {max_mismatch_val:.2e})")

    # Reconstruct final voltages (NumPy for final calculations)
    final_va = np.array(Va_init)
    final_vm = np.array(Vm_init)
    final_va[non_slack] = np.array(x[:len(non_slack)])
    final_vm[pq] = np.array(x[len(non_slack):])

    # ====================== FULL POWER FLOW RESULTS ======================
    final_V = final_vm * np.exp(1j * final_va)
    I_final = Ybus_np @ final_V
    S_final = final_V * np.conj(I_final)

    P_inj = np.real(S_final) * baseMVA
    Q_inj = np.imag(S_final) * baseMVA

    # Slack bus power
    slack_idx = next(i for i, bus in enumerate(bus_data) if bus.get('type', 1) == 3)
    slackP_MW = P_inj[slack_idx]
    slackQ_MVAr = Q_inj[slack_idx]

    # Branch power flows and losses
    branch_flow = []
    for branch in branch_data:
        f_idx = branch['fbus'] - 1
        t_idx = branch['tbus'] - 1
        r = branch['r']
        x = branch['x']
        b = branch['b']
        z = complex(r, x)
        if abs(z) < 1e-10:
            z = 1j * 1e-6
        y_ser = 1.0 / z
        b_half = 1j * b / 2.0

        Vf = final_V[f_idx]
        Vt = final_V[t_idx]
        Iij = (Vf - Vt) * y_ser + Vf * b_half
        Iji = (Vt - Vf) * y_ser + Vt * b_half

        Sij = Vf * np.conj(Iij)
        Sji = Vt * np.conj(Iji)

        Pij = np.real(Sij) * baseMVA
        Qij = np.imag(Sij) * baseMVA
        Pji = np.real(Sji) * baseMVA
        Qji = np.imag(Sji) * baseMVA
        Ploss = Pij + Pji

        branch_flow.append([branch['fbus'], branch['tbus'], Pij, Qij, Pji, Qji, Ploss])

    Ploss_total_MW = sum(row[6] for row in branch_flow)

    return {
        'Vm': final_vm,
        'Va_deg': np.rad2deg(final_va), # convert back to degrees for output
        'success': success,
        'iterations': iterations,
        'max_mismatch': max_mismatch_val,
        'P_inj_MW': P_inj,
        'Q_inj_MVAr': Q_inj,
        'slackP_MW': slackP_MW,
        'slackQ_MVAr': slackQ_MVAr,
        'Ploss_total_MW': Ploss_total_MW,
        'branch_flow': branch_flow   # list of [from, to, Pij, Qij, Pji, Qji, Ploss]
    }