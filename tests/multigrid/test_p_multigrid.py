import pytest
from firedrake import *

@pytest.fixture(params=["triangles", "quadrilaterals"], scope="module")
def mesh(request):
    if request.param == "triangles":
        base = UnitSquareMesh(2, 2)
        mh = MeshHierarchy(base, 1)
        mesh = mh[-1]
    elif request.param == "quadrilaterals":
        base = UnitSquareMesh(2, 2, quadrilateral=True)
        mh = MeshHierarchy(base, 1)
        mesh = mh[-1]
    return mesh

def test_p_multigrid_scalar(mesh):
    V = FunctionSpace(mesh, "CG", 4)

    u = Function(V)
    v = TestFunction(V)
    f = Constant(1)
    bcs = DirichletBC(V, 0, "on_boundary")

    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx

    relax = {"ksp_type": "chebyshev",
             "ksp_monitor_true_residual": None,
             "ksp_norm_type": "unpreconditioned",
             "ksp_max_it": 3,
             "pc_type": "jacobi"}

    sp = {"snes_monitor": None,
          "snes_type": "ksponly",
          "ksp_type": "fgmres",
          "ksp_monitor_true_residual": None,
          "pc_type": "python",
          "pc_python_type": "firedrake.PMGPC",
          "pmg_pc_mg_type": "multiplicative",
          "pmg_mg_levels": relax,
          "pmg_mg_coarse_ksp_type": "richardson",
          "pmg_mg_coarse_ksp_max_it": 1,
          "pmg_mg_coarse_ksp_norm_type": "unpreconditioned",
          "pmg_mg_coarse_ksp_monitor": None,
          "pmg_mg_coarse_pc_type": "mg",
          "pmg_mg_coarse_pc_mg_type": "multiplicative",
          "pmg_mg_coarse_mg_levels": relax,
          "pmg_mg_coarse_mg_coarse_ksp_type": "richardson",
          "pmg_mg_coarse_mg_coarse_ksp_max_it": 1,
          "pmg_mg_coarse_mg_coarse_ksp_norm_type": "unpreconditioned",
          "pmg_mg_coarse_mg_coarse_ksp_monitor": None,
          "pmg_mg_coarse_mg_coarse_pc_type": "gamg"}
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()

    assert solver.snes.ksp.its <= 5
    ppc = solver.snes.ksp.pc.getPythonContext().ppc
    assert ppc.getMGLevels() == 3
    assert ppc.getMGCoarseSolve().pc.getMGLevels() == 2


def test_p_multigrid_vector():
    mesh = UnitSquareMesh(2, 2)

    V = VectorFunctionSpace(mesh, "CG", 4)
    u = Function(V)

    rho = Constant(2700)
    g = Constant(-9.81)
    B = Constant((0.0, rho*g))  # Body force per unit volume

    # Kinematics
    I = Identity(2)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor
    E = 0.5 * (C - I)           # strain tensor

    # Elasticity parameters
    E_, nu = 6.9e10, 0.334
    mu, lmbda = Constant(E_/(2*(1 + nu))), Constant(E_*nu/((1 + nu)*(1 - 2*nu)))

    # Linear elastic energy
    E = 0.5 * (
                2*mu * inner(sym(grad(u)), sym(grad(u)))*dx  # noqa: E126
              + lmbda * inner(div(u), div(u))*dx             # noqa: E126
              - inner(B, u)*dx                               # noqa: E126
              )                                              # noqa: E126

    bcs = DirichletBC(V, zero((2,)), 1)

    F = derivative(E, u, TestFunction(V))
    sp = {"snes_monitor": None,
          "snes_type": "ksponly",
          "ksp_type": "fgmres",
          "ksp_rtol": 1.0e-8,
          "ksp_atol": 1.0e-8,
          "ksp_converged_reason": None,
          "ksp_monitor_true_residual": None,
          "pc_type": "python",
          "pc_python_type": "firedrake.PMGPC",
          "pmg_pc_mg_type": "full",
          "pmg_mg_levels_ksp_type": "chebyshev",
          "pmg_mg_levels_ksp_monitor_true_residual": None,
          "pmg_mg_levels_ksp_norm_type": "unpreconditioned",
          "pmg_mg_levels_ksp_max_it": 2,
          "pmg_mg_levels_pc_type": "pbjacobi",
          "pmg_mg_coarse_ksp_type": "richardson",
          "pmg_mg_coarse_ksp_max_it": 1,
          "pmg_mg_coarse_ksp_norm_type": "unpreconditioned",
          "pmg_mg_coarse_ksp_monitor": None,
          "pmg_mg_coarse_pc_type": "lu"}
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()

    assert solver.snes.ksp.its <= 20
    assert solver.snes.ksp.pc.getPythonContext().ppc.getMGLevels() == 3
