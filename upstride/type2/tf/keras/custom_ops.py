from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops
upstride_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_upstride_ops.so'))

# these 6 functions are for optimization of linear layers. These are python versions for debugging


def upstride_inputs_py(a1, a2, a3, a4):
  ap_1 = a4 + a2
  ap_2 = a1 - a3
  ap_3 = a1 + a3
  ap_4 = a4 - a2
  ap_5 = a4 - a3
  ap_6 = a2 + a1
  ap_7 = a1 - a2
  ap_8 = a4 + a3
  return ap_1, ap_2, ap_3, ap_4, ap_5, ap_6, ap_7, ap_8


def upstride_inputs_backprop_py(dap_1, dap_2, dap_3, dap_4, dap_5, dap_6, dap_7, dap_8):
  da1 = dap_2 + dap_3 + dap_6 + dap_7
  da2 = dap_1 - dap_4 + dap_6 - dap_7
  da3 = dap_3 - dap_2 - dap_5 + dap_8
  da4 = dap_1 + dap_4 + dap_5 + dap_8
  return da1, da2, da3, da4


def upstride_kernels_py(b1, b2, b3, b4):
  bp_1 = b2 + b3
  bp_2 = b1 + b4
  bp_3 = b1 - b4
  bp_4 = b2 - b3
  bp_5 = b3 - b4
  bp_6 = b2 + b1
  bp_7 = b3 + b4
  bp_8 = b1 - b2
  return bp_1, bp_2, bp_3, bp_4, bp_5, bp_6, bp_7, bp_8


def upstride_kernels_backprop_py(dbp_1, dbp_2, dbp_3, dbp_4, dbp_5, dbp_6, dbp_7, dbp_8):
  db1 = dbp_2 + dbp_3 + dbp_6 + dbp_8
  db2 = dbp_1 + dbp_4 + dbp_6 - dbp_8
  db3 = dbp_1 - dbp_4 + dbp_5 + dbp_7
  db4 = dbp_2 - dbp_3 - dbp_5 + dbp_7
  return db1, db2, db3, db4


def upstride_outputs_py(cp_1, cp_2, cp_3, cp_4, cp_5, cp_6, cp_7, cp_8):
  A_2 = cp_1 + cp_2 + cp_3
  A_5 = (A_2 + cp_4) / 2
  c_1 = A_5 - cp_1 + cp_5
  c_2 = A_5 - A_2 + cp_6
  c_3 = A_5 - cp_2 + cp_7
  c_4 = A_5 - cp_3 + cp_8
  return c_1, c_2, c_3, c_4


def upstride_outputs_backprop_py(dc_1, dc_2, dc_3, dc_4):
  t1 = dc_1 + dc_2
  t2 = dc_3 + dc_4
  t3 = dc_1 - dc_2
  t4 = dc_3 - dc_4
  dcp_1 = .5 * (t2 - t1)
  dcp_2 = .5 * (t3 - t4)
  dcp_3 = .5 * (t3 + t4)
  dcp_4 = .5 * (t1 + t2)
  dcp_5 = dc_1
  dcp_6 = dc_2
  dcp_7 = dc_3
  dcp_8 = dc_4
  return dcp_1, dcp_2, dcp_3, dcp_4, dcp_5, dcp_6, dcp_7, dcp_8

# now these are cpp versions


def upstride_inputs(a1, a2, a3, a4):
  output = upstride_ops.upstride_input(a1, a2, a3, a4)
  return [output[i] for i in range(8)] 


def upstride_kernels(b1, b2, b3, b4):
  output = upstride_ops.upstride_kernel(b1, b2, b3, b4)
  return [output[i] for i in range(8)] 


def upstride_outputs(cp_1, cp_2, cp_3, cp_4, cp_5, cp_6, cp_7, cp_8):
  output = upstride_ops.upstride_output(cp_1, cp_2, cp_3, cp_4, cp_5, cp_6, cp_7, cp_8)
  return [output[i] for i in range(4)] 


@ops.RegisterGradient("UpstrideInput")
def _upstride_input_grad(op, *grads):
  return upstride_ops.upstride_input_grad(*grads)


@ops.RegisterGradient("UpstrideKernel")
def _upstride_kernel_grad(op, *grads):
  return upstride_ops.upstride_kernel_grad(*grads)


@ops.RegisterGradient("UpstrideOutput")
def _upstride_output_grad(op, *grads):
  return upstride_ops.upstride_output_grad(*grads)
