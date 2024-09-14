from torch._inductor.codegen import cpp, wrapper
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V


class ExtensionWrapperCodegen(wrapper.WrapperCodeGen):
    def __init__(self):
        super().__init__()


class ExtensionScheduling(BaseScheduling):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._scheduling = cpp.CppScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        return True

    def can_fuse_horizontal(self, node1, node2):
        return True

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_template(self, template_node, epilogue_nodes):
        pass

    def codegen_nodes(self, nodes):
        self._scheduling.codegen_nodes(nodes)

    def codegen_sync(self):
        pass

    def flush(self):
        self._scheduling.flush()
