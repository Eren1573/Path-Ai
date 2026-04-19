import pygame
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

WIDTH = 800
HEIGHT = 600

ROAD_LEFT = 200
ROAD_WIDTH = 400
LANE_WIDTH = ROAD_WIDTH // 3

LANES = [
    ROAD_LEFT + LANE_WIDTH//2,
    ROAD_LEFT + LANE_WIDTH + LANE_WIDTH//2,
    ROAD_LEFT + 2*LANE_WIDTH + LANE_WIDTH//2
]

pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("PATH AI Autonomous Driving")
clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial",22)

# ===============================
# DQN
# ===============================

class DQN(nn.Module):

    def __init__(self,input_size,output_size):

        super(DQN,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,output_size)
        )

    def forward(self,x):
        return self.net(x)

# ===============================
# AGENT
# ===============================

class Agent:

    def __init__(self,state_size,action_size):

        self.model = DQN(state_size,action_size)
        self.target = DQN(state_size,action_size)

        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(),lr=0.001)
        self.loss = nn.MSELoss()

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.memory = []

    def act(self,state):

        if random.random() < self.epsilon:
            return random.randint(0,2)

        s = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q = self.model(s)

        return torch.argmax(q).item()

    def remember(self,s,a,r,s2,d):

        self.memory.append((s,a,r,s2,d))

        if len(self.memory) > 5000:
            self.memory.pop(0)

    def train(self):

        if len(self.memory) < 64:
            return

        batch = random.sample(self.memory,64)

        s = torch.FloatTensor(np.array([b[0] for b in batch]))
        a = torch.LongTensor([b[1] for b in batch]).unsqueeze(1)
        r = torch.FloatTensor([b[2] for b in batch]).unsqueeze(1)
        s2 = torch.FloatTensor(np.array([b[3] for b in batch]))
        d = torch.FloatTensor([b[4] for b in batch]).unsqueeze(1)

        q = self.model(s).gather(1,a)

        with torch.no_grad():
            maxq = self.target(s2).max(1)[0].unsqueeze(1)

        target = r + (1-d)*self.gamma*maxq

        loss = self.loss(q,target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ===============================
# AI CAR
# ===============================

class AICar:

    def __init__(self):

        self.lane = 1
        self.x = LANES[self.lane]
        self.target_x = self.x
        self.y = HEIGHT-120
        self.speed = 5

    def update(self):

        if self.x < self.target_x:
            self.x += 5

        if self.x > self.target_x:
            self.x -= 5

    def change_lane(self,new_lane):

        if new_lane >=0 and new_lane <=2:
            self.lane = new_lane
            self.target_x = LANES[new_lane]

    def draw(self):

        pygame.draw.rect(screen,(255,0,0),(self.x-20,self.y,40,60))

    def lidar(self,vehicles,animals):

        rays = []
        angles = [-45,-20,0,20,45]

        for a in angles:

            length = 0
            hit = False

            while length < 200:

                length += 5

                x = self.x + math.sin(math.radians(a))*length
                y = self.y - math.cos(math.radians(a))*length

                for v in vehicles:
                    if abs(x-v.x) < 20 and abs(y-v.y) < 40:
                        hit = True

                for an in animals:
                    if an.active and abs(x-an.x) < 20 and abs(y-an.y) < 20:
                        hit = True

                if hit:
                    break

            rays.append(length)

            pygame.draw.line(screen,(0,255,0),(self.x,self.y),(x,y),1)

        return np.array(rays)/200

# ===============================
# VEHICLE
# ===============================

class Vehicle:

    def __init__(self):

        self.lane = random.randint(0,2)
        self.x = LANES[self.lane]
        self.y = random.randint(-600,-50)
        self.speed = random.randint(3,6)

    def update(self):

        self.y += self.speed

        if self.y > HEIGHT:
            self.y = random.randint(-600,-50)
            self.lane = random.randint(0,2)
            self.x = LANES[self.lane]

    def draw(self):

        pygame.draw.rect(screen,(0,0,255),(self.x-20,self.y,40,60))

# ===============================
# ANIMAL (LESS FREQUENT)
# ===============================

class Animal:

    def __init__(self):

        self.x = -100
        self.y = -100
        self.active = False
        self.spawn_timer = random.randint(200,500)

    def spawn(self):

        self.y = random.randint(100,400)

        if random.random() < 0.5:
            self.x = ROAD_LEFT
            self.dir = 1
        else:
            self.x = ROAD_LEFT+ROAD_WIDTH
            self.dir = -1

        self.active = True

    def update(self):

        if not self.active:

            self.spawn_timer -= 1

            if self.spawn_timer <= 0:

                if random.random() < 0.3:
                    self.spawn()

                self.spawn_timer = random.randint(200,500)

            return

        self.x += self.dir*3

        if self.x < ROAD_LEFT-40 or self.x > ROAD_LEFT+ROAD_WIDTH+40:

            self.active = False
            self.spawn_timer = random.randint(200,500)

    def draw(self):

        if self.active:
            pygame.draw.rect(screen,(139,69,19),(self.x,self.y,30,30))

# ===============================
# ROAD
# ===============================

def draw_road():

    pygame.draw.rect(screen,(60,60,60),(ROAD_LEFT,0,ROAD_WIDTH,HEIGHT))

    for i in range(1,3):

        x = ROAD_LEFT + i*LANE_WIDTH

        pygame.draw.line(screen,(255,255,255),(x,0),(x,HEIGHT),5)

# ===============================
# COLLISION
# ===============================

def collision(car,vehicles,animals):

    rect = pygame.Rect(car.x-20,car.y,40,60)

    for v in vehicles:

        if rect.colliderect(pygame.Rect(v.x-20,v.y,40,60)):
            return True

    for a in animals:

        if a.active and rect.colliderect(pygame.Rect(a.x,a.y,30,30)):
            return True

    return False

# ===============================
# INIT
# ===============================

car = AICar()

vehicles = [Vehicle() for _ in range(5)]
animals = [Animal() for _ in range(2)]

agent = Agent(5,3)

collisions = 0
distance = 0

# ===============================
# MAIN LOOP
# ===============================

running = True

while running:

    clock.tick(60)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    screen.fill((0,150,0))

    draw_road()

    state = car.lidar(vehicles,animals)

    action = agent.act(state)

    if action == 1:
        car.change_lane(car.lane-1)

    if action == 2:
        car.change_lane(car.lane+1)

    car.update()

    reward = 1

    for v in vehicles:
        v.update()
        v.draw()

    for a in animals:
        a.update()
        a.draw()

    if collision(car,vehicles,animals):

        reward = -100
        collisions += 1

    distance += car.speed

    next_state = car.lidar(vehicles,animals)

    agent.remember(state,action,reward,next_state,False)

    agent.train()

    car.draw()

    speed_text = font.render(f"Speed: {car.speed}",True,(255,255,255))
    col_text = font.render(f"Collisions: {collisions}",True,(255,255,255))
    dist_text = font.render(f"Distance: {distance}",True,(255,255,255))

    screen.blit(speed_text,(WIDTH-180,10))
    screen.blit(col_text,(WIDTH-180,35))
    screen.blit(dist_text,(WIDTH-180,60))

    pygame.display.update()

pygame.quit()