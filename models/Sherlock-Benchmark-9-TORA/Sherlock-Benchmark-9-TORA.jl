# # Translational Oscillations by a Rotational Actuator (TORA)
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/Sherlock-Benchmark-9-TORA.ipynb)

module TORA  #jl

using ClosedLoopReachability, MAT, Plots
import DifferentialEquations
using ClosedLoopReachability: UniformAdditivePostprocessing, NoSplitter

# The following option determines whether the verification settings should be
# used or not. The verification settings are chosen to show that the safety
# property is satisfied. Concretely we split the initial states into small
# chunks and run many analyses.
const verification = true;

# The following option determines whether the falsification settings should be
# used or not. The falsification settings are sufficient to show that the safety
# property is violated. Concretely we use a shorter time horizon.
const falsification = true;

# This model consists of a cart attached to a wall with a spring. The cart is
# free to move on a friction-less surface. The car has a weight attached to an
# arm, which is free to rotate about an axis. This serves as the control input
# to stabilize the cart at $x = 0$.
#
# ## Model
#
# The model is four dimensional, given by the following equations:
#
# ```math
# \left\{ \begin{array}{lcl}
#       \dot{x}_1 &=& x_2 \\
#       \dot{x}_2 &=& -x_1 + 0.1 \sin(x_3) \\
#       \dot{x}_3 &=& x_4  \\
#       \dot{x}_4 &=& u
# \end{array} \right.
# ```
#
# A neural network controller was trained for this system. The trained network
# has 3 hidden layers, with 100 neurons in each layer (i.e., a total of 300
# neurons). Note that the output of the neural network $f(x)$ needs to be
# normalized in order to obtain $u$, namely $u = f(x) - 10$. The sampling time
# for this controller is 1s.

@taylorize function TORA!(dx, x, p, t)
    x₁, x₂, x₃, x₄, u = x

    aux = 0.1 * sin(x₃)
    dx[1] = x₂
    dx[2] = -x₁ + aux
    dx[3] = x₄
    dx[4] = u
    dx[5] = zero(u)
    return dx
end;

# We consider three types of controllers: a ReLU controller, a sigmoid
# controller, and a mixed relu/tanh controller.

path = @modelpath("Sherlock-Benchmark-9-TORA", "controllerTora.mat")
controller_relu = read_nnet_mat(path, act_key="act_fcns")

path = @modelpath("Sherlock-Benchmark-9-TORA", "nn_tora_sigmoid.mat")
controller_sigmoid = read_nnet_mat(path, act_key="act_fcns")

path = @modelpath("Sherlock-Benchmark-9-TORA", "nn_tora_relu_tanh.mat")
controller_relutanh = read_nnet_mat(path, act_key="act_fcns");

# ## Specification

# The verification problem is safety. For an initial set of $x_1 ∈ [0.6, 0.7]$,
# $x_2 ∈ [−0.7, −0.6]$, $x_3 ∈ [−0.4, −0.3]$, and $x_4 ∈ [0.5, 0.6]$, the
# system has to stay within the box $x ∈ [−2, 2]^4$ for a time window of 20s.

X₀ = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
U = ZeroSet(1)

vars_idx = Dict(:states=>1:4, :controls=>5)
ivp = @ivp(x' = TORA!(x), dim: 5, x(0) ∈ X₀ × U)

period = 1.0  # control period
control_postprocessing = UniformAdditivePostprocessing(-10.0)  # control postprocessing

problem(controller) = ControlledPlant(ivp, controller, vars_idx, period;
                                      postprocessing=control_postprocessing)

## Safety specification
T = 20.0  # time horizon
T_warmup = 2 * period  # shorter time horizon for dry run

safe_states = cartesian_product(BallInf(zeros(4), 2.0), Universe(1))
predicate_verification = X -> X ⊆ safe_states
predicate_falsification = sol -> any(isdisjoint(R, safe_states) for F in sol for R in F);

# ## Results

alg = TMJets(abstol=1e-10, orderT=8, orderQ=3)
alg_nn = DeepZ()

function benchmark(prob; T=T, splitter, goal::String, silent::Bool=false)
    ## We solve the controlled system:
    silent || println("flowpipe construction")
    res_sol = @timed sol = solve(prob, T=T, alg_nn=alg_nn, alg=alg,
                                 splitter=splitter)
    sol = res_sol.value
    silent || print_timed(res_sol)

    ## Next we check the property for an overapproximated flowpipe:
    silent || println("property checking")
    solz = overapproximate(sol, Zonotope)
    if goal == "verification"
        res_pred = @timed predicate_verification(solz)
        silent || print_timed(res_pred)
        if res_pred.value
            silent || println("The property is satisfied.")
        else
            silent || println("The property may be violated.")
        end
    elseif goal == "falsification"
        res_pred = @timed predicate_falsification(solz)
        silent || print_timed(res_pred)
        if res_pred.value
            silent || println("The property is violated.")
        else
            silent || println("The property may be satisfied.")
        end
    end
    return solz
end

function plot_helper(fig, sol, sim, vars, lab_sim, plot_X0)
    if vars[1] == 0
        safe_states_projected = project(safe_states, [vars[2]])
        time = Interval(0, T)
        safe_states_projected = cartesian_product(time, safe_states_projected)
    else
        safe_states_projected = project(safe_states, vars)
    end
    plot!(fig, safe_states_projected, color=:lightgreen, lab="safe states")
    if plot_X0 && 0 ∉ vars
        plot!(fig, project(X₀, vars), lab="X₀")
    end
    plot!(fig, sol, vars=vars, color=:yellow, lab="")
    plot_simulation!(fig, sim; vars=vars, color=:black, lab=lab_sim)
end

for case in 1:3
    if case == 1
        println("Running analysis with ReLU controller")
        prob = problem(controller_relu)
        scenario = "relu"
        T_reach = verification ? T : T_warmup  # shorter time horizon if not verifying
        splitter = verification ? BoxSplitter([4, 4, 3, 5]) : NoSplitter()
        trajectories = 10
        include_vertices = true
        lab_sim = ""
        plot_X0 = !verification
        goal = "verification"
        result = "verified"
    elseif case == 2
        println("Running analysis with sigmoid controller")
        prob = problem(controller_sigmoid)
        scenario = "sigmoid"
        T_reach = falsification ? 0.4 : T  # shorter time horizon if falsifying
        splitter = NoSplitter()
        trajectories = falsification ? 1 : 10
        include_vertices = !falsification
        lab_sim = falsification ? "simulation" : ""
        plot_X0 = !falsification
        goal = "falsification"
        result = "falsified"
    else
        println("Running analysis with ReLU/tanh controller")
        prob = problem(controller_relutanh)
        scenario = "relutanh"
        T_reach = falsification ? 0.4 : T  # shorter time horizon if falsifying
        splitter = NoSplitter()
        trajectories = falsification ? 1 : 10
        include_vertices = !falsification
        lab_sim = falsification ? "simulation" : ""
        plot_X0 = !falsification
        goal = "falsification"
        result = "falsified"
    end

    benchmark(prob; T=T_warmup, splitter=NoSplitter(), goal=goal, silent=true)  # warm-up
    res = @timed benchmark(prob; T=T_reach, splitter=splitter, goal=goal)  # benchmark
    sol = res.value
    println("total analysis time")
    print_timed(res);
    io = isdefined(Main, :io) ? Main.io : stdout
    print(io, "JuliaReach, TORA, $scenario, $result, $(res.time)\n")

    # We also compute some simulations:

    println("simulation")
    res = @timed simulate(prob, T=T_reach; trajectories=trajectories,
                          include_vertices=include_vertices)
    sim = res.value
    print_timed(res);

    # Finally we plot the results

    vars = (1, 2)
    fig = plot(xlab="x₁", ylab="x₂")
    plot_helper(fig, sol, sim, vars, lab_sim, plot_X0)
    savefig("TORA-$scenario-x1-x2.png")

    vars=(3, 4)
    fig = plot(xlab="x₃", ylab="x₄")
    plot_helper(fig, sol, sim, vars, lab_sim, plot_X0)
    savefig("TORA-$scenario-x3-x4.png")
end

end  #jl
nothing  #jl
