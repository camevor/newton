# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

import numpy as np
import warp as wp
import math
import newton
import newton.examples
from newton.geometry.utils import create_box_mesh, transform_points
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal, get_test_devices
from newton.utils import parse_usd
from newton.sim import joints

devices = get_test_devices()


class TestImportUsd(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_articulation(self):
        builder = newton.ModelBuilder()

        results = parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
            builder,
            collapse_fixed_joints=True,
        )
        self.assertEqual(builder.body_count, 9)
        self.assertEqual(builder.shape_count, 26)
        self.assertEqual(len(builder.shape_key), len(set(builder.shape_key)))
        self.assertEqual(len(builder.body_key), len(set(builder.body_key)))
        self.assertEqual(len(builder.joint_key), len(set(builder.joint_key)))
        # 8 joints + 1 free joint for the root body
        self.assertEqual(builder.joint_count, 9)
        self.assertEqual(builder.joint_dof_count, 14)
        self.assertEqual(builder.joint_coord_count, 15)
        self.assertEqual(builder.joint_type, [newton.JOINT_FREE] + [newton.JOINT_REVOLUTE] * 8)
        self.assertEqual(len(results["path_body_map"]), 9)
        self.assertEqual(len(results["path_shape_map"]), 26)

        collision_shapes = [
            i
            for i in range(builder.shape_count)
            if builder.shape_flags[i] & int(newton.geometry.SHAPE_FLAG_COLLIDE_SHAPES)
        ]
        self.assertEqual(len(collision_shapes), 13)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_articulation_no_visuals(self):
        builder = newton.ModelBuilder()

        results = parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
            builder,
            collapse_fixed_joints=True,
            load_non_physics_prims=False,
        )
        self.assertEqual(builder.body_count, 9)
        self.assertEqual(builder.shape_count, 13)
        self.assertEqual(len(builder.shape_key), len(set(builder.shape_key)))
        self.assertEqual(len(builder.body_key), len(set(builder.body_key)))
        self.assertEqual(len(builder.joint_key), len(set(builder.joint_key)))
        # 8 joints + 1 free joint for the root body
        self.assertEqual(builder.joint_count, 9)
        self.assertEqual(builder.joint_dof_count, 14)
        self.assertEqual(builder.joint_coord_count, 15)
        self.assertEqual(builder.joint_type, [newton.JOINT_FREE] + [newton.JOINT_REVOLUTE] * 8)
        self.assertEqual(len(results["path_body_map"]), 9)
        self.assertEqual(len(results["path_shape_map"]), 13)

        collision_shapes = [
            i
            for i in range(builder.shape_count)
            if builder.shape_flags[i] & int(newton.geometry.SHAPE_FLAG_COLLIDE_SHAPES)
        ]
        self.assertEqual(len(collision_shapes), 13)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_articulation_with_mesh(self):
        builder = newton.ModelBuilder()

        _ = parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "simple_articulation_with_mesh.usda"),
            builder,
            collapse_fixed_joints=True,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_joint_ordering(self):
        builder_dfs = newton.ModelBuilder()
        parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
            builder_dfs,
            collapse_fixed_joints=True,
            joint_ordering="dfs",
        )
        expected = [
            "front_left_leg",
            "front_left_foot",
            "front_right_leg",
            "front_right_foot",
            "left_back_leg",
            "left_back_foot",
            "right_back_leg",
            "right_back_foot",
        ]
        for i in range(8):
            self.assertTrue(builder_dfs.joint_key[i + 1].endswith(expected[i]))

        builder_bfs = newton.ModelBuilder()
        parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
            builder_bfs,
            collapse_fixed_joints=True,
            joint_ordering="bfs",
        )
        expected = [
            "front_left_leg",
            "front_right_leg",
            "left_back_leg",
            "right_back_leg",
            "front_left_foot",
            "front_right_foot",
            "left_back_foot",
            "right_back_foot",
        ]
        for i in range(8):
            self.assertTrue(builder_bfs.joint_key[i + 1].endswith(expected[i]))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_env_cloning(self):
        builder_no_cloning = newton.ModelBuilder()
        builder_cloning = newton.ModelBuilder()
        parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant_multi.usda"),
            builder_no_cloning,
            collapse_fixed_joints=True,
        )
        parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant_multi.usda"),
            builder_cloning,
            collapse_fixed_joints=True,
            cloned_env="/World/envs/env_0",
        )
        self.assertEqual(builder_cloning.articulation_key, builder_no_cloning.articulation_key)
        # ordering of the shape keys may differ
        shape_key_cloning = set(builder_cloning.shape_key)
        shape_key_no_cloning = set(builder_no_cloning.shape_key)
        self.assertEqual(len(shape_key_cloning), len(shape_key_no_cloning))
        for key in shape_key_cloning:
            self.assertIn(key, shape_key_no_cloning)
        self.assertEqual(builder_cloning.body_key, builder_no_cloning.body_key)
        # ignore keys that are not USD paths (e.g. "joint_0" gets repeated N times)
        joint_key_cloning = [k for k in builder_cloning.joint_key if k.startswith("/World")]
        joint_key_no_cloning = [k for k in builder_no_cloning.joint_key if k.startswith("/World")]
        self.assertEqual(joint_key_cloning, joint_key_no_cloning)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_mass_calculations(self):
        builder = newton.ModelBuilder()

        _ = parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
            builder,
            collapse_fixed_joints=True,
        )

        np.testing.assert_allclose(
            np.array(builder.body_mass),
            np.array(
                [
                    0.09677605,
                    0.00783155,
                    0.01351844,
                    0.00783155,
                    0.01351844,
                    0.00783155,
                    0.01351844,
                    0.00783155,
                    0.01351844,
                ]
            ),
            rtol=1e-5,
            atol=1e-7,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_cube_cylinder_joint_count(self):
        builder = newton.ModelBuilder()
        import_results = parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "cube_cylinder.usda"),
            builder,
            collapse_fixed_joints=True,
            invert_rotations=True,
        )
        self.assertEqual(builder.body_count, 1)
        self.assertEqual(builder.shape_count, 2)
        self.assertEqual(builder.joint_count, 1)

        usd_path_to_shape = import_results["path_shape_map"]
        expected = {
            "/World/Cylinder_dynamic/cylinder_reverse/mesh_0": {"mu": 0.2, "restitution": 0.3},
            "/World/Cube_static/cube2/mesh_0": {"mu": 0.75, "restitution": 0.3},
        }
        # Reverse mapping: shape index -> USD path
        shape_idx_to_usd_path = {v: k for k, v in usd_path_to_shape.items()}
        for shape_idx in range(builder.shape_count):
            usd_path = shape_idx_to_usd_path[shape_idx]
            if usd_path in expected:
                self.assertAlmostEqual(builder.shape_material_mu[shape_idx], expected[usd_path]["mu"], places=5)
                self.assertAlmostEqual(
                    builder.shape_material_restitution[shape_idx], expected[usd_path]["restitution"], places=5
                )

    def test_mesh_approximation(self):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        def box_mesh(scale=(1.0, 1.0, 1.0), transform: wp.transform | None = None):
            vertices, indices = create_box_mesh(scale)
            if transform is not None:
                vertices = transform_points(vertices, transform)
            return (vertices, indices)

        def create_collision_mesh(name, vertices, indices, approximation_method):
            mesh = UsdGeom.Mesh.Define(stage, name)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

            mesh.CreateFaceVertexCountsAttr().Set([3] * (len(indices) // 3))
            mesh.CreateFaceVertexIndicesAttr().Set(indices.tolist())
            mesh.CreatePointsAttr().Set([Gf.Vec3f(*p) for p in vertices.tolist()])
            mesh.CreateDoubleSidedAttr().Set(False)

            prim = mesh.GetPrim()
            meshColAPI = UsdPhysics.MeshCollisionAPI.Apply(prim)
            meshColAPI.GetApproximationAttr().Set(approximation_method)
            return prim

        def npsorted(x):
            return np.array(sorted(x))

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        self.assertTrue(stage)

        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        self.assertTrue(scene)

        scale = wp.vec3(1.0, 3.0, 0.2)
        tf = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity())
        vertices, indices = box_mesh(scale=scale, transform=tf)

        create_collision_mesh("/meshOriginal", vertices, indices, UsdPhysics.Tokens.none)
        create_collision_mesh("/meshConvexHull", vertices, indices, UsdPhysics.Tokens.convexHull)
        create_collision_mesh("/meshBoundingSphere", vertices, indices, UsdPhysics.Tokens.boundingSphere)
        create_collision_mesh("/meshBoundingCube", vertices, indices, UsdPhysics.Tokens.boundingCube)

        builder = newton.ModelBuilder()
        newton.geometry.MESH_MAXHULLVERT = 4
        parse_usd(
            stage,
            builder,
        )

        self.assertEqual(builder.body_count, 0)
        self.assertEqual(builder.shape_count, 4)
        self.assertEqual(
            builder.shape_type, [newton.GeoType.MESH, newton.GeoType.MESH, newton.GeoType.SPHERE, newton.GeoType.BOX]
        )

        # original mesh
        mesh_original = builder.shape_source[0]
        self.assertEqual(mesh_original.vertices.shape, (8, 3))
        assert_np_equal(mesh_original.vertices, vertices)
        assert_np_equal(mesh_original.indices, indices)

        # convex hull
        mesh_convex_hull = builder.shape_source[1]
        self.assertEqual(mesh_convex_hull.vertices.shape, (4, 3))

        # bounding sphere
        self.assertIsNone(builder.shape_source[2])
        self.assertEqual(builder.shape_type[2], newton.geometry.GeoType.SPHERE)
        self.assertAlmostEqual(builder.shape_scale[2][0], wp.length(scale))
        assert_np_equal(np.array(builder.shape_transform[2].p), np.array(tf.p), tol=1.0e-4)

        # bounding box
        assert_np_equal(npsorted(builder.shape_scale[3]), npsorted(scale), tol=1.0e-6)
        # only compare the position since the rotation is not guaranteed to be the same
        assert_np_equal(np.array(builder.shape_transform[3].p), np.array(tf.p), tol=1.0e-4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_visual_match_collision_shapes(self):
        builder = newton.ModelBuilder()
        parse_usd(
            newton.examples.get_asset("humanoid.usda"),
            builder,
        )
        self.assertEqual(builder.shape_count, 38)
        self.assertEqual(builder.body_count, 16)
        visual_shape_keys = [k for k in builder.shape_key if "visuals" in k]
        collision_shape_keys = [k for k in builder.shape_key if "collisions" in k]
        self.assertEqual(len(visual_shape_keys), 19)
        self.assertEqual(len(collision_shape_keys), 19)
        visual_shapes = [i for i, k in enumerate(builder.shape_key) if "visuals" in k]
        # corresponding collision shapes
        collision_shapes = [builder.shape_key.index(k.replace("visuals", "collisions")) for k in visual_shape_keys]
        # ensure that the visual and collision shapes match
        for i in range(len(visual_shapes)):
            vi = visual_shapes[i]
            ci = collision_shapes[i]
            self.assertEqual(builder.shape_type[vi], builder.shape_type[ci])
            self.assertEqual(builder.shape_source[vi], builder.shape_source[ci])
            assert_np_equal(np.array(builder.shape_transform[vi]), np.array(builder.shape_transform[ci]), tol=1e-5)
            assert_np_equal(np.array(builder.shape_scale[vi]), np.array(builder.shape_scale[ci]), tol=1e-5)
            self.assertFalse(builder.shape_flags[vi] & int(newton.geometry.SHAPE_FLAG_COLLIDE_SHAPES))
            self.assertTrue(builder.shape_flags[ci] & int(newton.geometry.SHAPE_FLAG_COLLIDE_SHAPES))


class TestImportSampleAssets(unittest.TestCase):
    def verify_usdphysics_parser(self, file, model):
        """Verify model based on the UsdPhysics Parsing Utils"""
        # [1] https://openusd.org/release/api/usd_physics_page_front.html
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.Open(file)
        parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, ["/"])
        body_key_to_idx = dict(zip(model.body_key, range(model.body_count)))
        shape_key_to_idx = dict(zip(model.shape_key, range(model.shape_count)))

        parsed_bodies = list(zip(*parsed[UsdPhysics.ObjectType.RigidBody]))

        # body presence
        for body_path, _ in parsed_bodies:
            assert body_key_to_idx.get(str(body_path), None) is not None
        self.assertEqual(len(parsed_bodies), model.body_count)

        # body colliders
        # TODO: exclude or handle bodies that have child shapes
        for body_path, body_desc in parsed_bodies:
            body_idx = body_key_to_idx.get(str(body_path), None)

            model_collisions = {model.shape_key[sk] for sk in model.body_shapes[body_idx]}
            parsed_collisions = {str(collider) for collider in body_desc.collisions}
            self.assertEqual(parsed_collisions, model_collisions)

        # body mass
        mass_verified = set()
        inertia_verified = set()

        body_mass = model.body_mass.numpy()
        body_inertia = model.body_inertia.numpy()
        # in newton, only rigid bodies have mass
        for body_path, body_desc in parsed_bodies:
            body_idx = body_key_to_idx.get(str(body_path), None)
            prim = stage.GetPrimAtPath(body_path)
            if prim.HasAPI(UsdPhysics.MassAPI):
                mass_api = UsdPhysics.MassAPI(prim)
                # Parents' explicit total masses override any mass properties specified further down in the subtree. [1]
                if mass_api.GetMassAttr().HasAuthoredValue():
                    mass = mass_api.GetMassAttr().Get()
                    self.assertAlmostEqual(body_mass[body_idx], mass, places=5)
                    mass_verified.add(body_idx)
                if mass_api.GetDiagonalInertiaAttr().HasAuthoredValue():
                    diag_inertia = mass_api.GetDiagonalInertiaAttr().Get()
                    principal_axes = mass_api.GetPrincipalAxesAttr().Get().Normalize()
                    p = np.array(
                        wp.quat_to_matrix(wp.quat(*principal_axes.imaginary, principal_axes.real))
                    ).reshape((3, 3))
                    inertia = p.T @ np.diag(diag_inertia) @ p
                    assert_np_equal(body_inertia[body_idx], inertia, tol=1e-5)
                    inertia_verified.add(body_idx)

        # TODO: exclude or handle bodies that don't have explicit mass/inertia set
        self.assertEqual(len(mass_verified), model.body_count)
        self.assertEqual(len(inertia_verified), model.body_count)

        joint_mapping = {
            joints.JOINT_PRISMATIC: UsdPhysics.ObjectType.PrismaticJoint,
            joints.JOINT_REVOLUTE: UsdPhysics.ObjectType.RevoluteJoint,
            joints.JOINT_BALL: UsdPhysics.ObjectType.SphericalJoint,
            joints.JOINT_FIXED: UsdPhysics.ObjectType.FixedJoint,
            # joints.JOINT_FREE: None,
            joints.JOINT_DISTANCE: UsdPhysics.ObjectType.DistanceJoint,
            joints.JOINT_D6: UsdPhysics.ObjectType.D6Joint,
        }

        joint_key_to_idx = dict(zip(model.joint_key, range(model.joint_count)))
        model_joint_type = model.joint_type.numpy()
        joints_found = []

        for joint_type, joint_objtype in joint_mapping.items():
            for joint_path, joint_desc in list(zip(*parsed.get(joint_objtype, ()))):
                joint_idx = joint_key_to_idx.get(str(joint_path), None)
                joints_found.append(joint_idx)
                assert joint_key_to_idx.get(str(joint_path), None) is not None
                assert model_joint_type[joint_idx] == joint_type

        print(model.joint_key)
        self.assertEqual(len(joints_found) + 1, model.joint_count)

        body_q_array = model.body_q.numpy()
        joint_dof_dim_array = model.joint_dof_dim.numpy()
        body_positions = [body_q_array[i, 0:3].tolist() for i in range(body_q_array.shape[0])]
        body_quaternions = [body_q_array[i, 3:7].tolist() for i in range(body_q_array.shape[0])]

        total_dofs = 0
        for j in range(model.joint_count):
            lin = int(joint_dof_dim_array[j][0])
            ang = int(joint_dof_dim_array[j][1])
            total_dofs += lin + ang
            jt = int(model.joint_type.numpy()[j])
            
            if jt == newton.JOINT_REVOLUTE:
                self.assertEqual((lin, ang), (0, 1), f"{model.joint_key[j]} DOF dim mismatch")
            elif jt == newton.JOINT_FIXED:
                self.assertEqual((lin, ang), (0, 0), f"{model.joint_key[j]} DOF dim mismatch")
            elif jt == newton.JOINT_FREE:
                self.assertGreater(lin + ang, 0, f"{model.joint_key[j]} expected nonzero DOFs for free joint")
            elif jt == newton.JOINT_PRISMATIC:
                self.assertEqual((lin, ang), (1, 0), f"{model.joint_key[j]} DOF dim mismatch")
            elif jt == newton.JOINT_BALL:
                self.assertEqual((lin, ang), (0, 3), f"{model.joint_key[j]} DOF dim mismatch")
                
        self.assertEqual(int(total_dofs), int(model.joint_axis.numpy().shape[0]))
        joint_enabled = model.joint_enabled.numpy()
        self.assertTrue(all(joint_enabled[i] != 0 for i in range(len(joint_enabled))))

        axis_vectors = {
            "X": [1.0, 0.0, 0.0],
            "Y": [0.0, 1.0, 0.0],
            "Z": [0.0, 0.0, 1.0],
        }

        drive_gain_scale = 1.0
        scene = UsdPhysics.Scene.Get(stage, Sdf.Path("/physicsScene"))
        if scene:
            attr = scene.GetPrim().GetAttribute("warp:joint_drive_gains_scaling")
            if attr and attr.HasAuthoredValue():
                drive_gain_scale = float(attr.Get())

        for j, key in enumerate(model.joint_key):
            prim = stage.GetPrimAtPath(key)
            if not prim:
                continue
                
            dof_index = 0 if j <= 0 else sum(int(joint_dof_dim_array[i][0] + joint_dof_dim_array[i][1]) for i in range(j))
            
            p_rel = prim.GetRelationship("physics:body0")
            c_rel = prim.GetRelationship("physics:body1")
            p_targets = p_rel.GetTargets() if p_rel and p_rel.HasAuthoredTargets() else []
            c_targets = c_rel.GetTargets() if c_rel and c_rel.HasAuthoredTargets() else []
            
            if len(p_targets) == 1 and len(c_targets) == 1:
                p_path = str(p_targets[0])
                c_path = str(c_targets[0])
                if p_path in body_key_to_idx and c_path in body_key_to_idx:
                    self.assertEqual(int(model.joint_parent.numpy()[j]), body_key_to_idx[p_path])
                    self.assertEqual(int(model.joint_child.numpy()[j]), body_key_to_idx[c_path])
                    
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                axis_attr = prim.GetAttribute("physics:axis")
                axis_tok = axis_attr.Get() if axis_attr and axis_attr.HasAuthoredValue() else None
                if axis_tok:
                    expected_axis = axis_vectors[str(axis_tok)]
                    actual_axis = model.joint_axis.numpy()[dof_index].tolist()
                    
                    self.assertTrue(all(abs(actual_axis[i] - expected_axis[i]) < 1e-6 for i in range(3)) or 
                                  all(abs(actual_axis[i] - (-expected_axis[i])) < 1e-6 for i in range(3)))
                
                lower_attr = prim.GetAttribute("physics:lowerLimit")
                upper_attr = prim.GetAttribute("physics:upperLimit")
                lower = lower_attr.Get() if lower_attr and lower_attr.HasAuthoredValue() else None
                upper = upper_attr.Get() if upper_attr and upper_attr.HasAuthoredValue() else None
                
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    if lower is not None:
                        self.assertAlmostEqual(float(model.joint_limit_lower.numpy()[dof_index]), math.radians(lower), places=5)
                    if upper is not None:
                        self.assertAlmostEqual(float(model.joint_limit_upper.numpy()[dof_index]), math.radians(upper), places=5)
                else:
                    if lower is not None:
                        self.assertAlmostEqual(float(model.joint_limit_lower.numpy()[dof_index]), float(lower), places=5)
                    if upper is not None:
                        self.assertAlmostEqual(float(model.joint_limit_upper.numpy()[dof_index]), float(upper), places=5)
                        
            if prim.IsA(UsdPhysics.RevoluteJoint):
                ke_attr = prim.GetAttribute("drive:angular:physics:stiffness")
                kd_attr = prim.GetAttribute("drive:angular:physics:damping")
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                ke_attr = prim.GetAttribute("drive:linear:physics:stiffness")
                kd_attr = prim.GetAttribute("drive:linear:physics:damping")
            else:
                ke_attr = kd_attr = None
                
            if ke_attr:
                ke_val = ke_attr.Get() if ke_attr.HasAuthoredValue() else None
                if ke_val is not None:
                    ke = float(ke_val)
                    self.assertAlmostEqual(float(model.joint_target_ke.numpy()[dof_index]), ke * drive_gain_scale, places=5)
                    
            if kd_attr:
                kd_val = kd_attr.Get() if kd_attr.HasAuthoredValue() else None
                if kd_val is not None:
                    kd = float(kd_val)
                    self.assertAlmostEqual(float(model.joint_target_kd.numpy()[dof_index]), kd * drive_gain_scale, places=5)

        joint_X_p_array = model.joint_X_p.numpy()
        joint_X_c_array = model.joint_X_c.numpy()
        joint_X_p_positions = [joint_X_p_array[i, 0:3].tolist() for i in range(joint_X_p_array.shape[0])]
        joint_X_p_quaternions = [joint_X_p_array[i, 3:7].tolist() for i in range(joint_X_p_array.shape[0])]
        joint_X_c_positions = [joint_X_c_array[i, 0:3].tolist() for i in range(joint_X_c_array.shape[0])]
        joint_X_c_quaternions = [joint_X_c_array[i, 3:7].tolist() for i in range(joint_X_c_array.shape[0])]

        for j in range(model.joint_count):
            p = int(model.joint_parent.numpy()[j])
            c = int(model.joint_child.numpy()[j])
            if p < 0 or c < 0:
                continue
                
            parent_tf = wp.transform(wp.vec3(*body_positions[p]), wp.quat(*body_quaternions[p]))
            child_tf = wp.transform(wp.vec3(*body_positions[c]), wp.quat(*body_quaternions[c]))
            joint_parent_tf = wp.transform(wp.vec3(*joint_X_p_positions[j]), wp.quat(*joint_X_p_quaternions[j]))
            joint_child_tf = wp.transform(wp.vec3(*joint_X_c_positions[j]), wp.quat(*joint_X_c_quaternions[j]))
            
            lhs_tf = wp.transform_multiply(parent_tf, joint_parent_tf)
            rhs_tf = wp.transform_multiply(child_tf, joint_child_tf)
            
            lhs_p = wp.transform_get_translation(lhs_tf)
            rhs_p = wp.transform_get_translation(rhs_tf)
            lhs_q = wp.transform_get_rotation(lhs_tf)
            rhs_q = wp.transform_get_rotation(rhs_tf)
            
            self.assertTrue(all(abs(lhs_p[i] - rhs_p[i]) < 1e-6 for i in range(3)))
            
            q_diff = lhs_q * wp.quat_inverse(rhs_q)
            angle_diff = 2.0 * math.acos(min(1.0, abs(q_diff[3])))
            self.assertLessEqual(angle_diff, 1e-3)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_g1(self):
        builder = newton.ModelBuilder()
        asset_path = "/home/chris/Documents/usd-test-assets/isaaclab/g1.usd"

        newton.utils.parse_usd(
            asset_path,
            builder,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            load_non_physics_prims=False,
        )
        model = builder.finalize()
        self.verify_usdphysics_parser(asset_path, model)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_anymal(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        asset_root = newton.utils.download_asset("anymal_usd")
        stage_path = None
        for root, _, files in os.walk(asset_root):
            if "anymal_d.usda" in files:
                stage_path = os.path.join(root, "anymal_d.usda")
                break
        if not stage_path or not os.path.exists(stage_path):
            raise unittest.SkipTest(f"Stage file not found: {stage_path}")

        newton.utils.parse_usd(
            stage_path,
            builder,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            load_non_physics_prims=False,
        )
        model = builder.finalize()
        self.verify_usdphysics_parser(stage_path, model)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
