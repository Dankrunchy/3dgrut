# %% [markdown]
# ## Depict

# %%
import torch
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.basic_camera import BasicCamera

from omegaconf import OmegaConf

from tqdm import tqdm

config = OmegaConf.load("configs/base_experiment.yaml")

cloud: MixtureOfGaussians = MixtureOfGaussians(config)

# # thick tensor
# cloud.set_gaussian(
#     torch.tensor([0.0, 0.0, 0.0],       dtype=torch.float32, device="cuda"),
#     torch.tensor([0.3, 1.0, 0.3],       dtype=torch.float32, device="cuda"),
#     torch.tensor([1.0, 0.0, 0.0, 0.0],  dtype=torch.float32, device="cuda"),
#     torch.tensor([1.0, 1.0, 1.0],       dtype=torch.float32, device="cuda"),
#     torch.tensor([0.5],                 dtype=torch.float32, device="cuda")
# )

# thin tensor
cloud.set_gaussian(
    torch.tensor([0.0, 0.0, 0.0],       dtype=torch.float32, device="cuda"),
    torch.tensor([0.3, 0.001, 0.3],     dtype=torch.float32, device="cuda"),
    torch.tensor([1.0, 0.0, 0.0, 0.0],  dtype=torch.float32, device="cuda"),
    torch.tensor([1.0, 1.0, 1.0],       dtype=torch.float32, device="cuda"),
    torch.tensor([0.5],                 dtype=torch.float32, device="cuda")
)

camera = BasicCamera(
    cloud, config,
    5, 5, 
    torch.tensor([0,-10,0], dtype=torch.float32, device="cuda"),
    torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
    10
)

# %%
img: NDArray = camera.render().detach().cpu().numpy()

plt.imshow(img)
plt.show()

# %% [markdown]
# ## Plot Error Rate

# %%
import torch
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.basic_camera import BasicCamera

from omegaconf import OmegaConf

from tqdm import tqdm

config = OmegaConf.load("configs/base_experiment.yaml")

cloud: MixtureOfGaussians = MixtureOfGaussians(config, 1.0)

# # zero rotation big gaussian
# cloud.add_gaussian(
#     MU.float3(0, 0, 0),
#     MU.float3(3, 1, 3),
#     Quaternion.normalize( Quaternion(1, 0, 0, 0) ),
#     MU.float3(1,1,1),
#     0.5
# )

# # zero rotation
# cloud.add_gaussian(
#     MU.float3(0, 0, 0),
#     MU.float3(0.03, 1, 0.03),
#     Quaternion.normalize( Quaternion(1, 0, 0, 0) ),
#     MU.float3(1,1,1),
#     0.5
# )
# cloud.set_gaussian(
#     torch.tensor([0.0, 0.0, 0.0],       dtype=torch.float32, device="cuda"),
#     torch.tensor([0.03, 1, 0.03],       dtype=torch.float32, device="cuda"),
#     torch.tensor([1.0, 0.0, 0.0, 0.0],  dtype=torch.float32, device="cuda"),
#     torch.tensor([1.0, 1.0, 1.0],       dtype=torch.float32, device="cuda"),
#     torch.tensor([0.5],                 dtype=torch.float32, device="cuda")
# )

# # zero rotation
# cloud.add_gaussian(
#     MU.float3(0, 0, 0),
#     MU.float3(0.3, 1, 0.3),
#     Quaternion.normalize( Quaternion(1, 0, 0, 0) ),
#     MU.float3(1,1,1),
#     0.5
# )
cloud.set_gaussian(
    torch.tensor([0.0, 0.0, 0.0],       dtype=torch.float32, device="cuda"),
    torch.tensor([0.3, 1.0, 0.3],       dtype=torch.float32, device="cuda"),
    torch.tensor([1.0, 0.0, 0.0, 0.0],  dtype=torch.float32, device="cuda"),
    torch.tensor([1.0, 1.0, 1.0],       dtype=torch.float32, device="cuda"),
    torch.tensor([0.5],                 dtype=torch.float32, device="cuda")
)


# # 22.5° rotation
# cloud.add_gaussian(
#     MU.float3(0, 0, 0),
#     MU.float3(0.3, .1, 0.3),
#     Quaternion.normalize( Quaternion(0.9807853, 0, 0, 0.1950903) ),
#     MU.float3(1,1,1),
#     0.5
# )
# cloud.set_gaussian(
#     torch.tensor([0.0, 0.0, 0.0],       dtype=torch.float32, device="cuda"),
#     torch.tensor([0.3, 0.1, 0.3],       dtype=torch.float32, device="cuda"),
#     torch.tensor([0.9807853, 0, 0, 0.1950903],  dtype=torch.float32, device="cuda"),
#     torch.tensor([1.0, 1.0, 1.0],       dtype=torch.float32, device="cuda"),
#     torch.tensor([0.5],                 dtype=torch.float32, device="cuda")
# )

# # 45° rotation
# cloud.add_gaussian(
#     MU.float3(0, 0, 0),
#     MU.float3(.4, .1, .3),
#     Quaternion.normalize( Quaternion(0.9238795, 0, 0, 0.3826834) ),
#     MU.float3(1,1,1),
#     0.5
# )
# cloud.set_gaussian(
#     torch.tensor([0.0, 0.0, 0.0],       dtype=torch.float32, device="cuda"),
#     torch.tensor([0.3, 1.0, 0.3],       dtype=torch.float32, device="cuda"),
#     torch.tensor([0.9238795, 0, 0, 0.3826834],  dtype=torch.float32, device="cuda"),
#     torch.tensor([1.0, 1.0, 1.0],       dtype=torch.float32, device="cuda"),
#     torch.tensor([0.5],                 dtype=torch.float32, device="cuda")
# )

# # 67.5° rotation
# cloud.add_gaussian(
#     MU.float3(0, 0, 0),
#     MU.float3(.7, .1, .3), 
#     Quaternion.normalize( Quaternion(0.8314696, 0, 0, 0.5555702) ),
#     MU.float3(1,1,1),
#     0.5
# )

# # 89° rotation
# cloud.add_gaussian(
#     MU.float3(0, 0, 0),
#     MU.float3(1, .1, .3),
#     Quaternion.normalize( Quaternion(0.7132504, 0, 0, 0.7009093) ),
#     MU.float3(1,1,1),
#     0.5
# )
# cloud.set_gaussian(
#     torch.tensor([0.0, 0.0, 0.0],       dtype=torch.float32, device="cuda"),
#     torch.tensor([1.0, 0.1, 0.3],       dtype=torch.float32, device="cuda"),
#     torch.tensor([0.7132504, 0, 0, 0.7009093],  dtype=torch.float32, device="cuda"),
#     torch.tensor([1.0, 1.0, 1.0],       dtype=torch.float32, device="cuda"),
#     torch.tensor([0.5],                 dtype=torch.float32, device="cuda")
# )

cam = BasicCamera(
    cloud, config,
    5, 5, 
    torch.tensor([0,-10,0], dtype=torch.float32, device="cuda"),
    torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
    10
)

render = cam.render()

# cloud.optimizer = torch.optim.Adam( [cloud.positions, cloud.scale, cloud.rotation] )

loss_fn = torch.nn.functional.l1_loss

# one offset to right from middle
target_image = torch.zeros( (cam.height, cam.width, 3), dtype=torch.float32, device="cuda" )
target_image[2, 2] = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32, device="cuda")

steps = 1_000 # 10_000
y_sizes = np.zeros(steps, dtype=np.float32)

pos_grads = np.zeros((steps,3), dtype=np.float32)
size_grads = np.zeros((steps, 3), dtype=np.float32)
quaternion_grads = np.zeros((steps, 4), dtype=np.float32)

losses = np.zeros((steps), dtype=np.float32)

_sub_st = steps // 10
y_sizes[:_sub_st] = 1 / (10 ** np.linspace(0, 1, _sub_st))       # from 1.0 to 0.1
y_sizes[_sub_st:] = 10 ** -np.linspace(1, 6, steps-_sub_st) # from 0.1 to 1e-5

# _distances = np.zeros(steps, dtype=float)

cloud.train(True)

plt.imshow(target_image.detach().cpu().numpy())

for i, y_size in tqdm( enumerate( list( y_sizes ) ) ):
    print(i)
    # update size
    with torch.no_grad():
        cloud.scale[0, 1] = cloud.scale_activation_inv( torch.tensor(y_size) ).cuda()
    # cloud.set_gaussian(
    #     torch.tensor([0.0, 0.0, 0.0],       dtype=torch.float32, device="cuda"),
    #     torch.tensor([0.3, y_size, 0.3],    dtype=torch.float32, device="cuda"),
    #     torch.tensor([1.0, 0.0, 0.0, 0.0],  dtype=torch.float32, device="cuda"),
    #     torch.tensor([1.0, 1.0, 1.0],       dtype=torch.float32, device="cuda"),
    #     torch.tensor([0.5],                 dtype=torch.float32, device="cuda")
    # )

    render = cam.render(target_image)

    if i > 100 and False:
        plt.imshow(render.detach().cpu().numpy())
        plt.show()

    loss = loss_fn( render, target_image )
    assert loss.item() != 0
    losses[i] = loss.detach().item()
    loss.backward()

    # positive y gradient means away from camera
    pos_grads[i] = cloud.positions.grad[0].detach().cpu().numpy()

    # assert np.sum(pos_grads[i]) == 0.0

    size_grads[i] = cloud.scale.grad[0].detach().cpu().numpy()
    
    quaternion_grads[i] = cloud.rotation.grad[0].detach().cpu().numpy()
    
    cloud.positions.grad[...]   = 0
    cloud.scale.grad[...]       = 0
    cloud.rotation.grad[...]    = 0

# %%
import matplotlib.pyplot as plt

plt.imshow(render.detach().cpu().numpy())

f = plt.figure(figsize=(12, 8))

pos_grads_plot = pos_grads.copy()
pos_grads_plot = np.abs( pos_grads_plot )
# pos_grads_plot[ pos_grads != pos_grads ] = 1e12 # remove nans
pos_grads_plot[ pos_grads == 0 ] = 1e-12 # remove zeros (should not happen)

plt.plot( y_sizes, pos_grads_plot[:,0], label="x" )
plt.plot( y_sizes, pos_grads_plot[:,1], label="y" )
plt.plot( y_sizes, pos_grads_plot[:,2], label="z" )
plt.plot( y_sizes, losses, label="loss" )
plt.gca().invert_xaxis()    # from thick to thin
plt.legend()

plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Background grid
plt.xscale("log") # gradients can get very big, so use log scale
if (pos_grads_plot <= 0).any():
    plt.yscale("linear")
else:
    plt.yscale("log")
    
plt.title('Position Gradient dependent on Y-Scale. In theory this should be almost constant.')
plt.xlabel("Y-Scale")
plt.ylabel('Pos Gradient')

# from pathlib import Path
# path = Path("/home/user1/Documents/Relightable3DGaussian/tmp")
# plt.savefig( str(path / f"numerical_position_issues_GS.png") )
plt.show()
# plt.close()

# %%
import matplotlib.pyplot as plt

f = plt.figure(figsize=(12, 8))

size_grads_plot = size_grads.copy()
# size_grads_plot = np.abs( size_grads_plot )
# size_grads_plot[ size_grads != size_grads ] = 1e2 # remove nans
size_grads_plot[ size_grads == 0 ] = 1e-12 # remove zeros (should not happen)

plt.plot( y_sizes, size_grads_plot[:,0], label="x" )
plt.plot( y_sizes, size_grads_plot[:,1], label="y" )
plt.plot( y_sizes, size_grads_plot[:,2], label="z" )
plt.gca().invert_xaxis()    # from thick to thin
plt.legend()

plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Background grid
plt.xscale("log") # gradients can get very big, so use log scale
if (size_grads_plot <= 0).any():
    plt.yscale("linear")
else:
    plt.yscale("log")
    
plt.title('Size Gradient dependent on Y-Scale. In theory this should be constant after y gets small enough')
plt.xlabel("Y-Scale")
plt.ylabel('Size Gradient')

from pathlib import Path
path = Path("/home/user1/Documents/Master/worktree/gaussian_renderer/tmp")
# plt.savefig( str(path / f"numerical_issues_solved.png") )
plt.show()
# plt.close()

# %%
import matplotlib.pyplot as plt

f = plt.figure(figsize=(12, 8))

quaternion_grads_plot = quaternion_grads.copy()
# quaternion_grads_plot = np.abs( quaternion_grads_plot )
# quaternion_grads_plot[ quaternion_grads != quaternion_grads ] = 1e2 # remove nans
# quaternion_grads_plot[ quaternion_grads == 0 ] = 1e-14 # remove zeros (should not happen)

plt.plot( y_sizes, quaternion_grads_plot[:,0], label="w" )
plt.plot( y_sizes, quaternion_grads_plot[:,1], label="x" )
plt.plot( y_sizes, quaternion_grads_plot[:,2], label="y" )
plt.plot( y_sizes, quaternion_grads_plot[:,3], label="z" )
plt.gca().invert_xaxis()    # from thick to thin
plt.legend()

plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Background grid
plt.xscale("log") # gradients can get very big, so use log scale
if (quaternion_grads_plot <= 0).any():
    plt.yscale("linear")
else:
    plt.yscale("log")
    
plt.title('Quaternion Gradient dependent on Y-Scale. I don\'t know what this should look like, but probably constant after a while.')
plt.xlabel("Y-Scale")
plt.ylabel('Quaternion Gradient')

from pathlib import Path
path = Path("/home/user1/Documents/Master/worktree/gaussian_renderer/tmp")
# plt.savefig( str(path / f"numerical_issues_solved.png") )
plt.show()
# plt.close()

# %%



