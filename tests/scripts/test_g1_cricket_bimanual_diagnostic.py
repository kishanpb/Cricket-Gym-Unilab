"""Constraint-force interpretation and immutable diagnostic outputs."""

import mujoco
import numpy as np
import pytest
from retarget_g1_cricket_batting import grip_force, main


def test_connect_force_norm_and_inactive_rows():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep=".001" gravity="0 0 0"/>
      <worldbody>
        <site name="anchor" pos="0 0 1"/>
        <body pos="0 0 1"><freejoint/>
          <geom type="sphere" size=".05" mass="1" contype="0" conaffinity="0"/>
          <site name="grip"/>
        </body>
      </worldbody>
      <equality><connect site1="grip" site2="anchor" solref=".012 1"/></equality>
    </mujoco>""")
    data = mujoco.MjData(model)
    data.qfrc_applied[0] = 2
    for _ in range(300):
        mujoco.mj_step(model, data)
    assert grip_force(data, 0) == pytest.approx(2, abs=1e-5)
    constraint = np.empty(model.nv)
    mujoco.mj_mulJacTVec(model, data, constraint, data.efc_force)
    np.testing.assert_allclose(constraint, data.qfrc_constraint, atol=1e-10)
    assert constraint[0] < 0
    data.eq_active[0] = 0
    mujoco.mj_forward(model, data)
    with pytest.raises(RuntimeError, match="three constraint rows"):
        grip_force(data, 0)


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["retarget", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        main()
    assert not list(tmp_path.iterdir())
