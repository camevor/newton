# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

from newton import Model, ModelBuilder
from newton.tests.test_builder_replicate import BuilderMergeTestCase

# Clone provenance only exists on the clone_builders() destination, so it is excluded
# when comparing against a manual begin_world()/add_builder()/end_world() expansion.
CLONE_METADATA_ATTRIBUTES = (
    "prototype_label",
    "prototype_counts",
    "clone_prototype",
    "clone_label",
    "clone_starts",
)


def _xform(x: float) -> wp.transform:
    return wp.transform((x, 0.0, 0.5 * x), wp.quat_rpy(0.0, 0.05 * x, 0.1))


class TestModelBuilderClone(BuilderMergeTestCase):
    @staticmethod
    def _make_second_source() -> ModelBuilder:
        builder = ModelBuilder()
        body = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), label="crate")
        joint = builder.add_joint_free(parent=-1, child=body, label="crate_root")
        builder.add_articulation([joint], label="crate_articulation")
        builder.add_shape_box(body=body, hx=0.2, hy=0.2, hz=0.2, label="crate_shape")
        builder.add_shape_sphere(
            body=-1,
            radius=0.1,
            xform=wp.transform((0.0, 1.0, 0.0), wp.quat_identity()),
            label="crate_static",
        )
        return builder

    @staticmethod
    def _make_plan():
        world_builders = [[0, 1], [1], [0, 0]]
        world_xforms = [[_xform(1.0), _xform(2.0)], [_xform(3.0)], [_xform(4.0), _xform(5.0)]]
        return world_builders, world_xforms

    def test_clone_builders_matches_manual_expansion(self):
        world_builders, world_xforms = self._make_plan()
        for use_prefixes in (False, True):
            with self.subTest(use_prefixes=use_prefixes):
                sources = [self._make_source(), self._make_second_source()]
                world_prefixes = ["front", "middle", "back"] if use_prefixes else None
                prototype_labels = ["robot", "crate"] if use_prefixes else None

                # Resolved clone labels: world prefix joined with the prototype label
                # (generated as prototype_{id} when prototype_labels is omitted).
                resolved_prototype_labels = prototype_labels or ["prototype_0", "prototype_1"]
                expected = self._make_destination()
                for r, row in enumerate(world_builders):
                    expected.begin_world()
                    for k, slot in enumerate(row):
                        parts = (world_prefixes[r] if world_prefixes else None, resolved_prototype_labels[slot])
                        expected.add_builder(
                            sources[slot],
                            xform=world_xforms[r][k],
                            label_prefix="/".join(part for part in parts if part),
                        )
                    expected.end_world()

                actual = self._make_destination()
                actual.clone_builders(
                    sources,
                    world_builders,
                    world_xforms=world_xforms,
                    world_prefixes=world_prefixes,
                    prototype_labels=prototype_labels,
                )

                self.assert_builder_merge_state_equal(expected, actual, ignore=CLONE_METADATA_ATTRIBUTES)

    def test_batched_matches_sequential_executor(self):
        world_builders, world_xforms = self._make_plan()
        for use_coord_layout_targets in (False, True):
            with (
                self.subTest(use_coord_layout_targets=use_coord_layout_targets),
                mock.patch("newton.use_coord_layout_targets", use_coord_layout_targets),
            ):
                sources = [self._make_source(), self._make_second_source()]

                batched = self._make_destination()
                batched.clone_builders(
                    sources,
                    world_builders,
                    world_xforms=world_xforms,
                    world_prefixes=["w0", "w1", "w2"],
                    prototype_labels=["robot", "crate"],
                    clone_prefixes=[["robot_a", "crate_a"], ["crate_b"], ["robot_b", "robot_c"]],
                )

                sequential = self._make_destination()
                plan = sequential._prepare_clone_plan(
                    sources,
                    world_builders,
                    world_xforms,
                    ["w0", "w1", "w2"],
                    ["robot", "crate"],
                    [["robot_a", "crate_a"], ["crate_b"], ["robot_b", "robot_c"]],
                )
                sequential._execute_clone_plan_sequential(plan)

                self.assert_builder_merge_state_equal(sequential, batched)

    def test_clone_metadata_contents(self):
        robot = self._make_source()
        crate = self._make_second_source()
        destination = self._make_destination()
        base_bodies = destination.body_count

        destination.clone_builders(
            [robot, crate],
            [[0, 1], [1, 0]],
            prototype_labels=["robot", "crate"],
        )

        self.assertEqual(destination.prototype_label, ["robot", "crate"])
        self.assertEqual(destination.clone_prototype, [0, 1, 1, 0])
        self.assertEqual(destination.clone_label, ["robot", "crate", "crate", "robot"])

        body_frequency = Model.AttributeFrequency.BODY
        shape_frequency = Model.AttributeFrequency.SHAPE
        self.assertEqual(destination.prototype_counts[body_frequency], [robot.body_count, crate.body_count])
        self.assertEqual(destination.prototype_counts[shape_frequency], [robot.shape_count, crate.shape_count])
        expected_body_starts = [
            base_bodies,
            base_bodies + robot.body_count,
            base_bodies + robot.body_count + crate.body_count,
            base_bodies + robot.body_count + 2 * crate.body_count,
        ]
        self.assertEqual(destination.clone_starts[body_frequency], expected_body_starts)

        # Clone ID plus prototype-local index resolves to the prefixed destination label,
        # and the half-open range covers exactly the prototype's entities.
        for clone_id, prototype in enumerate(destination.clone_prototype):
            source = [robot, crate][prototype]
            start = destination.clone_starts[shape_frequency][clone_id]
            count = destination.prototype_counts[shape_frequency][prototype]
            self.assertEqual(count, source.shape_count)
            clone_label = destination.clone_label[clone_id]
            for local, label in enumerate(source.shape_label):
                expected_label = f"{clone_label}/{label}" if label else label
                self.assertEqual(destination.shape_label[start + local], expected_label)

        # The clone's world is recoverable from any cloned entity.
        self.assertEqual(
            [destination.body_world[start] for start in destination.clone_starts[body_frequency]],
            [1, 1, 2, 2],
        )

    def test_unreferenced_prototype_merges_no_state(self):
        robot = self._make_source()
        unreferenced = self._make_second_source()
        unreferenced.request_contact_attributes("force")
        unreferenced.request_state_attributes("body_qdd")

        destination = ModelBuilder()
        destination.clone_builders([robot, unreferenced], [[0]])

        # The unreferenced slot registers a prototype but must not merge any state,
        # matching the sequential expansion that never adds it.
        self.assertEqual(destination.prototype_label, ["prototype_0", "prototype_1"])
        self.assertEqual(destination.clone_prototype, [0])
        self.assertEqual(destination._requested_contact_attributes, set())
        self.assertEqual(destination._requested_state_attributes, set())

    def test_repeated_calls_extend_metadata(self):
        robot = self._make_source()
        crate = self._make_second_source()
        destination = ModelBuilder()

        destination.clone_builders([robot], [[0]])
        first_labels = list(destination.clone_label)
        self.assertEqual(destination.prototype_label, ["prototype_0"])
        self.assertEqual(first_labels, ["prototype_0"])

        # A reused source builder defines a new prototype; unreferenced slots still
        # allocate prototype IDs; existing clone IDs remain stable.
        destination.clone_builders([crate, robot], [[1]])
        self.assertEqual(destination.prototype_label, ["prototype_0", "prototype_1", "prototype_2"])
        self.assertEqual(destination.clone_prototype, [0, 2])
        self.assertEqual(destination.clone_label[: len(first_labels)], first_labels)
        self.assertEqual(destination.world_count, 2)

        body_frequency = Model.AttributeFrequency.BODY
        self.assertEqual(
            destination.clone_starts[body_frequency],
            [0, robot.body_count],
        )

    def test_precondition_validation(self):
        robot = self._make_source()
        crate = self._make_second_source()

        finalized = self._make_second_source()
        finalized.begin_world()
        finalized.end_world()

        cases = {
            "empty builders": ([], [[0]], {}),
            "empty world_builders": ([robot], [], {}),
            "empty row": ([robot], [[0], []], {}),
            "bool index": ([robot], [[True]], {}),
            "float index": ([robot], [[0.0]], {}),
            "negative index": ([robot], [[-1]], {}),
            "out-of-range index": ([robot, crate], [[2]], {}),
            "source with worlds": ([finalized], [[0]], {}),
            "empty source": ([ModelBuilder()], [[0]], {}),
            "world_xforms outer shape": ([robot], [[0]], {"world_xforms": []}),
            "world_xforms inner shape": ([robot], [[0]], {"world_xforms": [[_xform(1.0), _xform(2.0)]]}),
            "world_xforms inner None": ([robot], [[0]], {"world_xforms": [[None]]}),
            "world_prefixes length": ([robot], [[0]], {"world_prefixes": ["a", "b"]}),
            "world_prefixes empty string": ([robot], [[0]], {"world_prefixes": [""]}),
            "prototype_labels length": ([robot], [[0]], {"prototype_labels": ["a", "b"]}),
            "prototype_labels empty string": ([robot], [[0]], {"prototype_labels": [""]}),
            "clone_prefixes shape": ([robot], [[0, 0]], {"clone_prefixes": [["a"]]}),
            "clone_prefixes empty string": ([robot], [[0]], {"clone_prefixes": [[""]]}),
        }
        for name, (builders, world_builders, kwargs) in cases.items():
            with self.subTest(case=name):
                destination = self._make_destination()
                untouched = self._make_destination()
                with self.assertRaises(ValueError):
                    destination.clone_builders(builders, world_builders, **kwargs)
                self.assert_builder_merge_state_equal(untouched, destination)

        destination = self._make_destination()
        with self.assertRaises(ValueError):
            destination.clone_builders([robot, destination], [[0, 1]])

        destination = self._make_destination()
        destination.begin_world()
        with self.assertRaises(RuntimeError):
            destination.clone_builders([robot], [[0]])

    def test_collapse_fixed_joints_raises_after_cloning(self):
        destination = ModelBuilder()
        destination.clone_builders([self._make_source()], [[0]])
        snapshot = ModelBuilder()
        snapshot.clone_builders([self._make_source()], [[0]])
        with self.assertRaises(RuntimeError):
            destination.collapse_fixed_joints()
        self.assert_builder_merge_state_equal(snapshot, destination)

    def test_gravity_mismatch_warns_and_last_builder_wins(self):
        default_gravity = self._make_second_source()
        zero_gravity = self._make_second_source()
        zero_gravity.gravity = (0.0, 0.0, 0.0)

        destination = ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "different gravity"):
            destination.clone_builders([default_gravity, zero_gravity], [[0, 1]])
        np.testing.assert_allclose(np.asarray(destination.world_gravity[0]), (0.0, 0.0, 0.0))

        destination = ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            destination.clone_builders([default_gravity, zero_gravity], [[1], [0]])
        self.assertEqual([str(w.message) for w in caught], [])
        np.testing.assert_allclose(np.asarray(destination.world_gravity[0]), (0.0, 0.0, 0.0))
        np.testing.assert_allclose(np.asarray(destination.world_gravity[1]), (0.0, 0.0, -9.81))

    @staticmethod
    def _add_scalar_attribute(builder: ModelBuilder, frequency: Model.AttributeFrequency, value: int | None) -> None:
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tag",
                dtype=wp.int32,
                frequency=frequency,
                namespace="test",
                default=-1,
            )
        )
        if value is not None:
            builder.custom_attributes["test:tag"].values = {0: value}

    def test_conflicting_once_values_warn_and_last_wins(self):
        for frequency, pattern in (
            (Model.AttributeFrequency.ONCE, "ONCE"),
            (Model.AttributeFrequency.WORLD, "WORLD"),
        ):
            with self.subTest(frequency=frequency.name):
                first = self._make_second_source()
                second = self._make_second_source()
                self._add_scalar_attribute(first, frequency, 3)
                self._add_scalar_attribute(second, frequency, 7)

                destination = ModelBuilder()
                with self.assertWarnsRegex(UserWarning, f"conflicting explicit {pattern} values"):
                    destination.clone_builders([first, second], [[0, 1]])
                self.assertEqual(destination.custom_attributes["test:tag"].values[0], 7)

                # Values targeting different worlds do not conflict, and a declaration
                # without an explicit value does not conflict with an existing value.
                undeclared = self._make_second_source()
                self._add_scalar_attribute(undeclared, frequency, None)
                destination = ModelBuilder()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if frequency == Model.AttributeFrequency.WORLD:
                        destination.clone_builders([first, second, undeclared], [[0, 2], [1]])
                    else:
                        destination.clone_builders([first, undeclared], [[0, 1]])
                self.assertEqual([str(w.message) for w in caught], [])
                if frequency == Model.AttributeFrequency.WORLD:
                    self.assertEqual(destination.custom_attributes["test:tag"].values, {0: 3, 1: 7})
                else:
                    self.assertEqual(destination.custom_attributes["test:tag"].values, {0: 3})

    def test_conflicting_reference_values_compared_after_mapping(self):
        # Both prototypes reference their body 0; after mapping into destination
        # indices the values differ, which must be reported despite equal raw values.
        first = self._make_second_source()
        second = self._make_second_source()
        for builder in (first, second):
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="root",
                    dtype=wp.int32,
                    frequency=Model.AttributeFrequency.ONCE,
                    namespace="test",
                    references="body",
                    default=-1,
                )
            )
            builder.custom_attributes["test:root"].values = {0: 0}

        destination = ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "conflicting explicit ONCE values"):
            destination.clone_builders([first, second], [[0, 1]])
        # Last clone wins with its mapped body index.
        self.assertEqual(destination.custom_attributes["test:root"].values[0], first.body_count)

    def test_domain_table_matches_merge_offset_map(self):
        builder = ModelBuilder()
        table_kinds = {
            ModelBuilder._builder_frequency_key(frequency) for frequency in ModelBuilder._CLONE_PROVENANCE_FREQUENCIES
        }
        self.assertEqual(table_kinds, set(ModelBuilder._builder_merge_counts(builder)))
        self.assertEqual(set(builder.prototype_counts), set(ModelBuilder._CLONE_PROVENANCE_FREQUENCIES))
        self.assertEqual(set(builder.clone_starts), set(ModelBuilder._CLONE_PROVENANCE_FREQUENCIES))

    def test_finalized_model_matches_expansion(self):
        sources = [self._make_source(), self._make_second_source()]
        world_builders, world_xforms = self._make_plan()
        destination = self._make_destination()
        destination.clone_builders(sources, world_builders, world_xforms=world_xforms)
        model = destination.finalize()
        self.assertEqual(model.world_count, 4)
        self.assertEqual(model.body_count, destination.body_count)
        body_world = model.body_world.numpy()
        body_frequency = Model.AttributeFrequency.BODY
        for clone_id in range(len(destination.clone_prototype)):
            start = destination.clone_starts[body_frequency][clone_id]
            count = destination.prototype_counts[body_frequency][destination.clone_prototype[clone_id]]
            worlds = set(body_world[start : start + count].tolist())
            self.assertEqual(len(worlds), 1)


if __name__ == "__main__":
    unittest.main()
