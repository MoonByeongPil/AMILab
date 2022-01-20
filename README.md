# AMILab

tf -> torch 명령어 모음
https://bladejun.tistory.com/146?category=481262

torch로만 바꿔주면 되는 목록
where
ones_like
reshape
square
sqrt
multiply



주의할 함수들
tf.tile(input, multiples, name=None)
torch_tensor.repeat((num, ...)

tf.reduce_mean
torch.mean --> 이렇게 하면 torch(z,3)이라 했을 때 z가 정수로 이루어진 tensor면 오류남. floating pt이거나 complex dtype이어야 한다.

tf.reduce_sum
torch.sum

tf.is_nan
torch.isnan

tf.truediv
torch.div --> truediv 없음. 둘 다 출력 tensor

tf.argsort(values, axis=-1, direction='ASCENDING')
torch.argsort(input, dim=-1, descending=False) 

tf.arg_max(input, dimension, output_type=tf.dtypes.int64, name=None)
torch.argmax(input)

tf.assign(ref, value, validate_shape=None, use_locking=None, name=None)

tf.boolean_mask(tensor, mask, name='boolean_mask', axis=No)
torch.masked_select(input, mask, *, out=None)

tf.cast(x, dtype, name=None)
Tensor_name.type

tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)
torch.clamp(input, min, max)

tf.concat(values, axis)
torch.cat

tf.fill(dims, value, name=None)
torch.full(size, fill_value)

tf.tile(input, multiples, name=None)
torch_tensor.repeat((num, ...)

tf.expand_dims( input, axis=None, name=None, dim=None)
torch.unsqueeze(input, dim)

tf.squeeze( input, axis=None, name=None, squeeze_dims=None)
torch.squeeze(input, dim)

tf.transpose(a)
torch.transpose(input, dim0, dim1)

tf.gather(params, indices, validate_indices=None, name=None, axis=None, batch_dims=0)
torch.gather(tensor, dim, indices)

tf.zeros(shape, dtype=tf.dtypes.float32, name=None)
torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

tf.ones_like(input, dtype=None, name=None)
torch.ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor

tf.math.abs(x, name=None)
torch.abs(x)

tf.math.cumsum(x, axis=0, exclusive=False, reverse=False, name=None)
torch.cumsum(input, dim, *, dtype=None, out=None) → Tensor


Loss function
optimzer
save
load
shape(?)