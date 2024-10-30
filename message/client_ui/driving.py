import pygame
import math
from client_api import ClientAPI
from render import Renderer
import pickle
import cv2
import numpy as np
from collections import deque

# Initialize Pygame
pygame.init()

# Initialize font module
pygame.font.init()
font = pygame.font.SysFont('Arial', 18)

# Screen dimensions in pixels
screen_width, screen_height = 800, 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Kinematic Bicycle Model Simulation")

# Clock object to control the simulation rate
clock = pygame.time.Clock()

# Physical dimensions in meters (simulation area)
physical_width = 160.0   # Width of the simulation area in meters
physical_height = 160.0   # Height of the simulation area in meters

# Scale factors to convert physical coordinates to screen pixels
scale_x = screen_width / physical_width
scale_y = screen_height / physical_height

# Vehicle parameters
L = 2.5  # Wheelbase of the vehicle in meters

# Fixed control inputs
acceleration = 5.0                      # Acceleration in m/s^2
max_steering_angle = math.radians(15)    # Maximum steering angle in radians

# Time step
dt = 1.0 / 10.0  # Time step for 10 Hz

# Initial state of the vehicle (physical coordinates)
x = physical_width / 2.0   # X-position in meters (centered)
y = physical_height / 2.0  # Y-position in meters (centered)
yaw = math.radians(90)                  # Heading angle in radians
v = 0.0                    # Velocity in m/s
delta = 0.0                # Steering angle in radians

# Main simulation loop flag
running = True
is_interactive = True
is_dump = True
hist_len = 6

if is_interactive:
    client = ClientAPI(host='10.40.11.68', port=8888)
    client.connect()

if is_interactive:
    anchor_dict = client.receive()
    if is_dump:
        pickle.dump(anchor_dict, open('anchor_dict.pkl', 'wb'))
else:
    anchor_dict = pickle.load(open('anchor_dict.pkl', 'rb'))
act_vocal = anchor_dict['traj']['veh']

local_render = Renderer(n_world=2, show_n_world=2, range=[80, 80])
count = 1000

# Initialize map_surface and ego_surface
map_surface = None
ego_surface = None
agent_surface = None


while running:
    if count > 80:
        if is_interactive:
            data = client.receive()
            if is_dump:
                pickle.dump(data, open('data.pkl', 'wb'))
        else:
            data = pickle.load(open('data.pkl', 'rb'))
        
        # Render map
        rendered_map = local_render._render_map(data['map_point_pos'][:, 0], 
                                               data['map_point_pos'][:, 1], 
                                               data['map_point_type'])
        # Convert to BGR for OpenCV and save
        rendered_map_bgr = cv2.cvtColor(rendered_map, cv2.COLOR_RGB2BGR)
        cv2.imwrite('rendered_map.png', rendered_map_bgr)

        # Transpose and convert to Pygame surface
        rendered_map = np.transpose(rendered_map, (1, 0, 2))
        map_surface = pygame.surfarray.make_surface(rendered_map)
        map_surface = pygame.transform.scale(map_surface, (screen_width, screen_height))
        
        count = 0
        ego_hist = deque(maxlen=hist_len)

    # Event handling loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get the state of all keyboard buttons
    keys = pygame.key.get_pressed()

    # Update control inputs based on key presses
    if keys[pygame.K_UP]:
        v += acceleration * dt          # Increase velocity
    if keys[pygame.K_DOWN]:
        v -= acceleration * dt          # Decrease velocity

    # Set steering angle directly based on key presses
    if keys[pygame.K_LEFT]:
        delta = max_steering_angle      # Set steering angle to max left
    elif keys[pygame.K_RIGHT]:
        delta = -max_steering_angle     # Set steering angle to max right
    else:
        delta = 0.0                     # No steering input

    # Blit the map_surface to screen if it exists
    if map_surface:
        screen.blit(map_surface, (0, 0))
    else:
        print("Map surface not found. Skipping rendering.")

    # Define the vehicle as a rectangle in physical dimensions (meters)

    if count == 0:
        av_index = data['av_index']
        yaw = data['heading'][av_index][0, 10]  
        ego_shape = data['shape'][av_index][0, 10]
        cur_xy = data['position'][av_index][0, 10, :2]
        prev_xy = data['position'][av_index][0, 5, :2]
        v = np.linalg.norm(cur_xy - prev_xy) / 0.5
        x, y, z = data['position'][av_index][0, 10]
        data['position'][av_index][0, 6:11]
        for i in range(11):
            x, y, z = data['position'][av_index][0, i]
            yaw = data['heading'][av_index][0, 10] 
            ego_hist.append((x, y, yaw)) 


    # Kinematic Bicycle Model equations to update vehicle state
    x += v * math.cos(yaw) * dt         # Update x-position in meters
    y += v * math.sin(yaw) * dt         # Update y-position in meters
    yaw += (v / L) * math.tan(delta) * dt  # Update heading angle

    ego_hist.append((x, y, yaw))

    # tranfer to ego centric coordinate
    ego_hist_np = np.array(ego_hist)
    refer_index = 0
    # Subtract the reference position
    ego_hist_np[:, :2] -= ego_hist_np[refer_index, :2]

    # Rotation matrix using the reference heading
    rotation_matrix = np.array([
        [np.cos(ego_hist_np[refer_index, 2]), np.sin(ego_hist_np[refer_index, 2])],
        [-np.sin(ego_hist_np[refer_index, 2]), np.cos(ego_hist_np[refer_index, 2])]
    ])

    # Apply rotation (this is where the issue is)
    ego_hist_np[:, :2] = np.dot(ego_hist_np[:, :2], rotation_matrix.T)
    ego_hist_np[:, 2] -= ego_hist_np[refer_index, 2]

    # import pdb; pdb.set_trace()
    diff = ego_hist_np[None, 0::5] - act_vocal[:, :2, :]

    xy_diff = np.linalg.norm(diff[..., :2], axis=2).mean(axis=1)
    angle_diff = np.abs(diff[..., 2]).mean(axis=-1)
    # import pdb; pdb.set_trace()
    select_act_index = np.argmin(xy_diff + angle_diff * 5)
    token_diff = xy_diff[select_act_index]
    select_act_traj = act_vocal[select_act_index, :, :]
    # import pdb; pdb.set_trace()
    # print(select_act_traj[..., :2])

    # Render ego vehicle
    ego_image = local_render._render_ego(x, y, yaw + np.pi/2, ego_shape, select_act_traj, ego_hist)
    ego_image = np.transpose(ego_image, (1, 0, 2))
    ego_surface = pygame.surfarray.make_surface(ego_image)
    ego_surface = pygame.transform.scale(ego_surface, (screen_width, screen_height))
    ego_surface.set_alpha(128)
    screen.blit(ego_surface, (0, 0))

    if count != 0 and count % 5 == 0:
        if is_interactive:
            client.send({'select_act_index': select_act_index, 'xyh': np.array([x,y,yaw])}) 
    if count == 80:
        count += 1
        continue  

    if count % 5 == 0:
        if is_interactive:  
            agent_data = client.receive()
            if is_dump:
                pickle.dump(agent_data, open('agent_data.pkl', 'wb'))
        else:
            agent_data = pickle.load(open('agent_data.pkl', 'rb'))
    
    position = agent_data['pred_traj'][:, count, :] # N, 2
    cat_av_mask = agent_data['cat_av_mask'] # N
    heading = np.tile(data['heading'], [2, 1])[:, count][..., None]
    shape = np.tile(data['shape'][:, 11], [2, 1])
    
    all_feat_agent = np.concatenate([position, heading, shape], axis=1)

    # Render agent image
    agent_image = local_render._render_agent(all_feat_agent, cat_av_mask)
    agent_image_surface = pygame.surfarray.make_surface(np.transpose(agent_image, (1, 0, 2)))
    agent_image_surface = pygame.transform.scale(agent_image_surface, (screen_width, screen_height))
    agent_image_surface.set_alpha(128)  # Adjust transparency as needed
    screen.blit(agent_image_surface, (0, 0))


    count += 1

    # Render kinematic values in the corner
    info_text = [
        f"Position: ({x:.2f}, {y:.2f}) m",
        f"Velocity: {v:.2f} m/s",
        f"Steering Angle: {math.degrees(delta):.2f} deg",
        f"Heading: {math.degrees(yaw)%360:.2f} deg",
        f"Token diff: {token_diff:.2f} m",
        f"Select act index: {select_act_index}",
        f"xy_diff: {xy_diff.min():.2f} m",
        f"ego_hist_np: {ego_hist_np[0::5, :2]}",
    ]

    # Position to start rendering text
    line_height = font.get_linesize()
    start_x = 10
    start_y = 10

    # Render each line of text
    for i, text in enumerate(info_text):
        text_surface = font.render(text, True, (0, 0, 0))
        screen.blit(text_surface, (start_x, start_y + i * line_height))

    # Update the display
    pygame.display.flip()

    # Control the simulation rate to run at 10 Hz
    clock.tick(10)

# Quit Pygame
pygame.quit()
