# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_simulator.compat.patches import PatchScope


def test_inherited_method_override_is_removed_on_scope_exit():
    class Base:
        def value(self):
            return 3

    class Derived(Base):
        pass

    with PatchScope() as scope:
        scope.replace(Derived, "value", lambda self: 7)
        assert Derived().value() == 7

    assert Derived().value() == 3
    assert "value" not in vars(Derived)


def test_staticmethod_descriptor_is_restored_without_binding_changes():
    class Owner:
        @staticmethod
        def value():
            return 3

    original = vars(Owner)["value"]
    with PatchScope() as scope:
        scope.replace(Owner, "value", staticmethod(lambda: 7))
        assert Owner.value() == 7

    assert vars(Owner)["value"] is original
    assert Owner.value() == 3
