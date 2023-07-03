import wgpu.utils
import wgpu.backends.rs
import numpy as np
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

class WGSLProgram:
  def __init__(self, name:str, prg:str):
    self.name, self.prg = name, prg

  def __call__(self, global_size, local_size, *args, wait=False):
    device = wgpu.utils.get_default_device()
    cshader = device.create_shader_module(code=self.prg.replace("WORKGROUP_SIZE", ','.join(map(str, local_size))))
    buffers = [device.create_buffer_with_data(data=arg._buffer(), usage=wgpu.BufferUsage.STORAGE | (wgpu.BufferUsage.COPY_SRC if i == 0 else 0)) for i, arg in enumerate(args)]

    # Setup layout and bindings
    binding_layouts = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {
      "type": wgpu.BufferBindingType.storage if i == 0 else wgpu.BufferBindingType.read_only_storage}} for i in range(len(args))]
    bindings = [{"binding": i, "resource": {"buffer": buf, "offset": 0, "size": buf.size}} for i, buf in enumerate(buffers)]

    # Put everything together
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

    # Create and run the pipeline
    compute_pipeline = device.create_compute_pipeline(
      layout=pipeline_layout,
      compute={"module": cshader, "entry_point": self.name},
    )
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
    compute_pass.dispatch_workgroups(*global_size)
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])

    # Read result
    args[0]._copyin(np.array(device.queue.read_buffer(buffers[0]).cast("f")))

class WGSLCodegen(CStyleCodegen):
  lang = CStyleLanguage(kernel_prefix = "@compute @workgroup_size(WORKGROUP_SIZE)",
                        wgsl_style = True,
                        gid = [f"gid.{chr(120+i)}" for i in range(3)], lid = [f"lid.{chr(120+i)}" for i in range(3)],
                        extra_args=['@builtin(workgroup_id) gid: vec3<u32>', '@builtin(local_invocation_id) lid: vec3<u32>'])
  supports_float4: bool = False

WGSLBuffer = Compiled(RawMallocBuffer, WGSLCodegen, WGSLProgram)
