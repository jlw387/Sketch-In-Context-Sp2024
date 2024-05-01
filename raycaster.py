"""
Main file for generating sphere-traced renders.
"""

from math import inf
import numpy as np
import torch 

from MegaNet.network import SketchToSDF, load_S2SDF

from dataset import load_image

from camera import *
from vector_math import *

from PIL import Image as im

import copy
from tqdm import tqdm

def find_t_range(o, d, s, eps=1e-6, print_long=False):
    t_values = []
    if abs(d[0]) >= eps:
        t_values.append((s - o[0])/d[0])
        t_values.append((-s - o[0])/d[0])
    if abs(d[1]) >= eps:
        t_values.append((s - o[1])/d[1])
        t_values.append((-s - o[1])/d[1])
    if abs(d[2]) >= eps:
        t_values.append((s - o[2])/d[2])
        t_values.append((-s - o[2])/d[2])

    t_sorted = np.sort(np.array(t_values))

    points = np.zeros((t_sorted.shape[0],3))
    for i in range(t_sorted.shape[0]):
        points[i] = o + d*t_sorted[i]
    
    if print_long:
        print("t values:", t_sorted)
        print("\nPoints:\n")
        for p in points:
            print(p)

    in_range = (np.abs(o + t_sorted[int((t_sorted.size-1)/2)] * d) <= s + eps).all() and (np.abs(o + t_sorted[int((t_sorted.size-1)/2) + 1]*d)<= s + eps).all()

    if print_long:   
        print("Min Check (Index = " + str(int((t_sorted.size-1)/2)) + "):", (np.abs(o + t_sorted[int((t_sorted.size-1)/2)] * d) <= s).all()) 
        print("Max Check (Index = " + str(int((t_sorted.size-1)/2) + 1) + "):", (np.abs(o + t_sorted[int((t_sorted.size-1)/2) + 1]*d)<= s).all())   

    if print_long:
        print("Check Points:")
        print(np.abs(o + t_sorted[int((t_sorted.size-1)/2)] * d))
        print(np.abs(o + t_sorted[int((t_sorted.size-1)/2) + 1]*d))
        print("\nCheck Bools:")
        print(np.abs(o + t_sorted[int((t_sorted.size-1)/2)] * d) <= s+1e-5)
        print(np.abs(o + t_sorted[int((t_sorted.size-1)/2) + 1]*d) <= s+1e-5)

    if print_long:
        print("=======================\n\n")
    return t_sorted[int((t_sorted.size-1)/2)], t_sorted[int((t_sorted.size-1)/2) + 1], in_range

def find_t_ranges(o, dirs, s, eps=1e-6, print_long=False):
    t_values = np.zeros((dirs.shape[0], dirs.shape[1], 6))
    t_values[:,:,0] = np.where(np.abs(dirs[:,:,0]) < eps, -inf, (-s - o[0])/ dirs[:,:,0])
    t_values[:,:,1] = np.where(np.abs(dirs[:,:,1]) < eps, -inf, (-s - o[1])/ dirs[:,:,1])
    t_values[:,:,2] = np.where(np.abs(dirs[:,:,2]) < eps, -inf, (-s - o[2])/ dirs[:,:,2])
    t_values[:,:,3] = np.where(np.abs(dirs[:,:,0]) < eps, inf, (s - o[0])/ dirs[:,:,0])
    t_values[:,:,4] = np.where(np.abs(dirs[:,:,1]) < eps, inf, (s - o[1])/ dirs[:,:,1])
    t_values[:,:,5] = np.where(np.abs(dirs[:,:,2]) < eps, inf, (s - o[2])/ dirs[:,:,2])
    
    t_sorted = np.sort(np.array(t_values), axis=-1)

    in_range = np.logical_and((np.abs(o + t_sorted[:,:,2].reshape(t_sorted.shape[0], t_sorted.shape[1],1) * dirs) <= s + eps).all(axis=-1), \
        (np.abs(o + t_sorted[:,:,3].reshape(t_sorted.shape[0], t_sorted.shape[1],1)*dirs) <= s + eps).all(axis=-1))

    return t_sorted[:,:,2], t_sorted[:,:,3], in_range


def cast_ray_network(camera, model : SketchToSDF, img, light_dir, ob_color, bg_color, timeout_color, max_iterations=1000, eps=1e-4, max_step_size=0.25):
    origin = camera.pos

    dirs = camera.get_all_ray_directions()
    if False:
        print("Find Range for " + str(row) + "," + str(col)+":\n=======================")
    # Need to be more careful with near/far raycast range to avoid points outside the training domain. 
    # This line restricts the ray's near/far range to the points that lie within a cube centered on the origin with half-size of the third parameter.
    nears, fars, irs = find_t_ranges(origin, dirs, 0.99, eps=1e-6, print_long=False)

    # nears = np.zeros_like(nears)
    # fars = np.ones_like(fars)*1000

    # print("(" + str(row) + "," + str(col) + ") passed \'In Range\' test")
    output = np.tile(bg_color, (camera.height, camera.width, 1))
    ob_color = ob_color.reshape((1,3))

    start_points = origin + nears.reshape(nears.shape[0], nears.shape[1], 1) * dirs
    
    img_tensor = img.unsqueeze(1)
    
    with torch.no_grad():
        z = model.forward_encoder_section(img_tensor)
    
    print("Img Tensor:", img_tensor.shape)
    print("Z:", z.shape)
    print("Points:", start_points.shape)

    current_points = start_points
    guesses = nears.reshape(nears.shape[0], nears.shape[1], 1)
    guesses_filter = np.ones_like(guesses).astype(bool) # has converged
    guesses_bisect = np.zeros_like(guesses).astype(bool)
    scale_factors = np.ones_like(guesses)
    inside_range = np.ones_like(nears).astype(bool)

    print("Guesses:", guesses.shape)
    print("Guesses Filter:", guesses_filter.shape)

    stored_evals = np.zeros((max_iterations, start_points.shape[0], start_points.shape[1]))
    last_iteration_pixels = np.zeros_like(nears).astype(bool)

    with torch.no_grad():
        result = model.forward_sdf_section(z, torch.Tensor(current_points)).cpu().unsqueeze(-1).numpy()
        evals = np.clip(result, -max_step_size, max_step_size)
        at_roots = np.abs(evals) < eps

        # if type(guess) != np.float64:
        #         print("Error!\nRow:",row,"\nCol:",col,"\nGuess:",guess,"\nType Of Guess:",type(guess),"\nPrev Guess:",prev_guess,"\nNear:",near,"\nFar:",far)

        prev_guesses = np.zeros(guesses.shape)
        prev_guesses2 = -1 * np.ones(guesses.shape)

        # eval_checks = []
        # guess_checks = []
        # bisect_checks = []
        # filter_checks = []
        # eval_checks2 = []

        for iteration in tqdm(range(max_iterations)):
            prev_guesses2 = prev_guesses
            prev_guesses = guesses
            guesses = np.where(guesses_filter, guesses + evals * scale_factors, guesses)
            # guesses_filter = np.logical_and(guesses_filter, guesses - prev_guesses2 < eps / 10)
            np.logical_and(guesses_filter, guesses - prev_guesses2 < eps, out=guesses_bisect)
            np.logical_and(guesses.reshape(guesses.shape[0], guesses.shape[1]) < fars,\
                           guesses.reshape(guesses.shape[0], guesses.shape[1]) > nears,\
                            out=inside_range)
            
            guesses[guesses_bisect] = (guesses[guesses_bisect] + prev_guesses[guesses_bisect]) / 2
            scale_factors[guesses_bisect] = 1/4
            
            stored_evals[iteration,:][inside_range] = evals.reshape(evals.shape[0], evals.shape[1]).copy()[inside_range]

            current_points = origin + guesses * dirs
            
            np.clip(model.forward_sdf_section(z, torch.Tensor(current_points)).cpu().unsqueeze(-1).numpy(), -max_step_size, max_step_size, out=evals)
            
            
            guesses_filter = np.abs(evals) > eps

            # eval_checks.append(copy.deepcopy(evals[76,86,0]))
            # guess_checks.append(copy.deepcopy(guesses[76,86,0]))
            # bisect_checks.append(copy.deepcopy(guesses_bisect[76,86,0]))
            # filter_checks.append(copy.deepcopy(guesses_filter[76,86,0]))
            # eval_checks2.append(evals[132, 174])

            
            if not np.logical_and(guesses_filter.reshape(guesses_filter.shape[0], guesses_filter.shape[1]), inside_range).any():
                break
            
            np.logical_and(guesses_filter.reshape(guesses_filter.shape[0], guesses_filter.shape[1]), inside_range, out=last_iteration_pixels)
            # if type(guess) != np.float64:
            #     print("Error!\nRow:",row,"\nCol:",col,"\nGuess:",guess,"\nType Of Guess:","\nPrev Guess:",prev_guess,"\nNear:",near,"\nFar:",far,"\nIterations:",iterations,"\nOrigin:",origin,"\nDirection:",direction)

            # if _ == 500:
            #     for i in range(128):
            #         for j in range(128):
            #             if guesses_filter[i,j] and inside_range[i,j]:
            #                 print("i:",i,"   j:",j)

    # np.savetxt("EvalsBisectScale.txt", np.array(eval_checks))
    # np.savetxt("GuessesBisectScale.txt", np.array(guess_checks))

    # plt.plot(eval_checks)
    # plt.show()
    # plt.plot(guess_checks)
    # plt.show()
    # plt.plot(bisect_checks)
    # plt.show()
    # plt.plot(eval_checks2)
    # plt.show()

    final_points = origin + guesses * dirs 
    # print(final_points.shape)  #(128, 3)
    # print(origin.shape)     #(3,)
    # print(dirs.shape)       #(128, 128, 3)
    # print(dirs[row,:].shape)  #(128, 3)
    # print("(" + str(row) + "," + str(col) + ") iterations:", iterations)
    # print("(" + str(row) + "," + str(col) + ") final guess:", guess)
    # print("(" + str(row) + "," + str(col) + ") final point:", current_point)
    # print("(" + str(row) + "," + str(col) + ") final signed distance:", val)

    object_hits = np.abs(evals) < eps
    object_hits = object_hits.reshape(object_hits.shape[0], object_hits.shape[1])
    print("Object Hits:", object_hits.shape)
    np.logical_and(guesses.reshape(guesses.shape[0], guesses.shape[1]) < fars,\
                   guesses.reshape(guesses.shape[0], guesses.shape[1]) > nears,\
                    out=inside_range)
    
    # inside_range = inside_range.reshape(inside_range.shape[0], inside_range.shape[1], 1)
    valid_hits = np.logical_and(object_hits, inside_range).astype(bool) 

    # print(valid_hits.shape)   #torch.Size([128])
    # print(valid_hits.sum())
    timeouts = np.logical_and(inside_range, np.logical_not(object_hits)).astype(bool)

    print("Final Points:", final_points.shape)
    print("Valid Hits:", valid_hits.shape)
    final_hits = final_points[valid_hits, :]
    final_hits = torch.autograd.Variable(torch.Tensor(final_hits), requires_grad=True)
    # print(final_points.shape)    #(128, 3)
    vals_surface = model.forward_sdf_section(z, final_hits)

    vals_surface.backward(torch.ones_like(vals_surface))

    grads = final_hits.grad
    print("Grads:", grads.shape)    #(128, 3)
    norms = grads.pow(2).sum(dim=1).sqrt().unsqueeze(1)
    grads = grads / norms
    print("Normed Grads:", grads.shape)

    if valid_hits.sum() > 0:
        # print(light_dir.shape, grads[valid_hits].shape)   #(3,) torch.Size([7, 3])
        dots = np.clip(np.einsum("i,ji->j",-light_dir, grads), 0, 1)
        # print(normed_grads[valid_hits])
        dots = dots.reshape(dots.shape[0], 1)
        
        color = ob_color * dots

        output[valid_hits, :] = color

    if timeouts.sum() > 0:
        output[timeouts, :] = timeout_color
        
    np.save("IterationBehavior.npy", stored_evals.reshape(stored_evals.shape[0], stored_evals.shape[1] * stored_evals.shape[2]))
    # output = np.where(valid_hits, np.tile(ob_color, (object_hits.shape[0], object_hits.shape[1], 1)), np.tile(bg_color, (object_hits.shape[0], object_hits.shape[1], 1)))
    
    return output.astype(np.uint8)

    # ob_color * max(np.dot(-light_dir, tf.reshape(grads/np.linalg.norm(grads), [-1])[-3:]), 0)

    # if np.abs(val) < eps:
    #     dot = max(np.dot(-light_dir, tf.reshape(grad/np.linalg.norm(grad), [-1])[-3:]), 0)
    #     return ob_color * dot
    # elif iterations == 1000:
    #     return np.array([0,64,0])
    # # elif guess > far or guess < near: 
    # #     return np.array([0,0,64])
    # else:
    #     return bg_color
# 2.828427 3.695518 1.5307337
camera_position = np.array([0,0,-4])
camera_fov = 30

light_direction = normalize(np.array([7,-10,4]))
object_color = np.array([128,112,255])
background_color = np.array([64,0,0])
timeout_color = np.array([0,64,0])

sketch_path = "./SDFDatasets/Gen_1712888438_Train/Model_0/sketch.png"
# network_timestamp = "2024_03_29_11_48_29"
#network_timestamp = "2022_08_16_11_17_16"
# network_timestamp = "2022_09_05_00_39_54"
network_timestamp = "2024_03_26_03_58_24"

model = load_S2SDF(network_dir="Models/SDF/" + network_timestamp, weights_string="FinalWeights")

image_size = 256
camera = Camera(width=image_size,height=image_size,fov=camera_fov, pos=camera_position, up=np.array([0,1,0]))

image_arr = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)

far_cube = 2 * np.linalg.norm(camera_position)

sketch_input = load_image(sketch_path)

image_arr = cast_ray_network(camera, model, sketch_input, light_direction, object_color, background_color, timeout_color)

image = im.fromarray(np.flip(image_arr, axis=1))
# image = im.fromarray(image_arr)

name = input("Enter image name: ")

if name.endswith(".png"):
    name = name[0:-4]

image.save(name + '.png')


