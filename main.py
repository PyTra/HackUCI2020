import cv2

def quickshow(name, frame):
  cv2.imshow(name, frame)
  cv2.waitKey(0)
  cv2.destroyWindow(name)

cap = cv2.VideoCapture(0)
while 1:
    _, im = cap.read()
    cv2.imshow('Aim Window', im)
    keypress = cv2.waitKey(1)
    if keypress == ord(' '):
        cap.release()
        break

blue_channel = im[:,:,0]
_, thresh = cv2.threshold(blue_channel, 80, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('Aim Window', cv2.drawContours(im.copy(), contours, -1, (255,0,0), 4 ) )
cv2.waitKey(0)
cv2.destroyWindow('Aim Window')

ptcoords = list(contours[index].T.reshape(2,-1) for index in range(len(contours) ) )
squarelist = list( 
    (ptcoords[index].min(axis = 1), 
     ptcoords[index].max(axis = 1) ) for index in range(len(ptcoords) ) )
if len(squarelist) == 0:
    squarelist = []
else:
    squarelist = squarelist[1:]
  
filtered_squarelist = list()
for ((minX, minY), (maxX, maxY)) in squarelist:
    if (maxX - minX) > 20 and (maxY - minY) > 20:
      filtered_squarelist.append( ( (minX, minY), (maxX, maxY) ) )
      quickshow('rect', cv2.rectangle(im.copy(), (minX, minY), (maxX, maxY), (255,0,0), thickness = 4 ) )



import cv2
import numpy as np
import pygame
from collections import defaultdict
from itertools import product
import time

pygame.init()
######################################################################
br1 = pygame.image.load("blueright (1).png")
br2 = pygame.image.load("blueright (2).png")
br7 = pygame.image.load("blueright (7).png")
br8 = pygame.image.load("blueright (8).png")
br26 = pygame.image.load("blueright (26).png")
br9 = pygame.image.load("blueright (9).png")
br4 = pygame.image.load("blueright (4).png")
br24 = pygame.image.load("blueright (24).png")
br11 = pygame.image.load("blueright (11).png")
br17 = pygame.image.load("blueright (17).png")
br18 = pygame.image.load("blueright (18).png")
br37 = pygame.image.load("blueright (37).png")
br38 = pygame.image.load("blueright (38).png")
br32 = pygame.image.load("blueright (32).png")
br34 = pygame.image.load("blueright (34).png")

rr1 = pygame.image.load("redright (2).png")
rr2 = pygame.image.load("redright (3).png")
rr7 = pygame.image.load("redright (8).png")
rr8 = pygame.image.load("redright (9).png")
rr26 = pygame.image.load("redright (27).png")
rr9 = pygame.image.load("redright (10).png")
rr4 = pygame.image.load("redright (33).png")
rr24 = pygame.image.load("redright (5).png")
rr11 = pygame.image.load("redright (12).png")
rr17 = pygame.image.load("redright (18).png")
rr18 = pygame.image.load("redright (19).png")
rr37 = pygame.image.load("redright (38).png")
rr38 = pygame.image.load("redright (39).png")
rr32 = pygame.image.load("redright (14).png")
rr34 = pygame.image.load("redright (35).png")

bl4 = pygame.image.load("blueleft (4).png")
bl5 = pygame.image.load("blueleft (5).png")
bl7 = pygame.image.load("blueleft (7).png")
bl8 = pygame.image.load("blueleft (8).png")
bl9 = pygame.image.load("blueleft (9).png")
bl33 = pygame.image.load("blueleft (33).png")
bl2 = pygame.image.load("blueleft (2).png")
bl14 = pygame.image.load("blueleft (14).png")
bl16 = pygame.image.load("blueleft (16).png")
bl17 = pygame.image.load("blueleft (17).png")
bl36 = pygame.image.load("blueleft (36).png")
bl39 = pygame.image.load("blueleft (39).png")
bl12 = pygame.image.load("blueleft (12).png")
bl10 = pygame.image.load("blueleft (10).png")

rl4 = pygame.image.load("redleft (52).png")
rl5 = pygame.image.load("redleft (58).png")
rl7 = pygame.image.load("redleft (61).png")
rl8 = pygame.image.load("redleft (62).png")
rl9 = pygame.image.load("redleft (11).png")
rl33 = pygame.image.load("redleft (16).png")
rl2 = pygame.image.load("redleft (56).png")
rl14 = pygame.image.load("redleft (68).png")
rl16 = pygame.image.load("redleft (70).png")
rl17 = pygame.image.load("redleft (71).png")
rl36 = pygame.image.load("redleft (19).png")
rl39 = pygame.image.load("redleft (22).png")
rl12 = pygame.image.load("redleft (66).png")
rl10 = pygame.image.load("redleft (14).png")

movement = {"Blue":
    {"Standing":
              {"Left": [[bl4, bl5], 0],
              "Right": [[br1, br2], 0]},
			"Walking":
              {"Left": [[bl7, bl8], 0],
              "Right": [[br7, br8, br26, br8], 0]},
      "Running":
              {"Left": bl9,
              "Right": br9},
      "Jumping":
              {"Left": bl33,
              "Right": br4},
      "Falling":
              {"Left": bl2,
              "Right": br24},
      "Idle Shooting":
              {"Left": bl14,
              "Right": br11},
      "Walking Shooting":
              {"Left": [[bl16, bl17], 0],
              "Right": [[br17, br18], 0]},
      "Running Shooting":
              {"Left": [[bl36, bl39], 0],
              "Right": [[br37, br38], 0]},
      "Jumping Shooting":
              {"Left": bl12,
              "Right": br32},
      "Falling Shooting":
              {"Left": bl10,
              "Right": br34}
			},
"Red":
      {"Standing":
              {"Left": [[rl4, rl5], 0],
              "Right": [[rr1, rr2], 0]},
			"Walking":
              {"Left": [[rl7, rl8], 0],
              "Right": [[rr7, rl8], 0]},
      "Running":
              {"Left": rl9,
              "Right": rr9},
      "Jumping":
              {"Left": rl33,
              "Right": rr4},
      "Falling":
              {"Left": rl2,
              "Right": rr24},
      "Idle Shooting":
              {"Left": rl14,
              "Right": rr11},
      "Walking Shooting":
              {"Left": [[rl16, rl17], 0],
              "Right": [[rr17, rr18], 0]},
      "Running Shooting":
              {"Left": [[rl36, rl39], 0],
              "Right": [[rr37, rr38], 0]},
      "Jumping Shooting":
              {"Left": rl12,
              "Right": rr32},
      "Falling Shooting":
              {"Left": rl10,
              "Right": rr34}
			}
}

def spriteEquivalent(team: int, position: list, velocity: list, aerial: bool, shooting: bool, orientation: str):
# BLUEEEEEEEEEEEEEEE
    #print('TEAM %s POSITION %s VELOCITY %s AERIAL %s SHOOTING %s ORIENTATION %s'%(team, position, velocity, aerial, shooting, orientation ) )
    shooting = not shooting
    if team == 0:
        if orientation == "left":
            if shooting:
                #Idle Left WITH GUN
                if velocity[0] == 0 and velocity[1] == 0:
                    return movement["Blue"]["Idle Shooting"]["Left"],position

                # Walking Left WITH GUN
                if -15 <= velocity[0] < 0 and velocity[1] == 0:
                    if movement["Blue"]["Walking Shooting"]["Left"][1] == len(movement["Blue"]["Walking Shooting"]["Left"][0]) - 1:
                        movement["Blue"]["Walking Shooting"]["Left"][1] = -1
                    movement["Blue"]["Walking Shooting"]["Left"][1] += 1
                    return movement["Blue"]["Walking Shooting"]["Left"][0][movement["Blue"]["Walking Shooting"]["Left"][1]],position
               
                # Running Left WITH GUN
                if -30 <= velocity[0] < -15 and velocity[1] == 0:
                    if movement["Blue"]["Running Shooting"]["Left"][1] == len(movement["Blue"]["Running Shooting"]["Left"][0]) - 1:
                        movement["Blue"]["Running Shooting"]["Left"][1] = -1
                    movement["Blue"]["Running Shooting"]["Left"][1] += 1
                    return movement["Blue"]["Running Shooting"]["Left"][0][movement["Blue"]["Running Shooting"]["Left"][1]],position
                
                # Jumping Left WITH GUN
                if velocity[1] > 0:
                    return movement["Blue"]["Jumping Shooting"]["Left"],position
               
                # Falling Left WITH GUN
                if velocity[1] < 0:
                    return movement["Blue"]["Falling Shooting"]["Left"],position
                  
                if velocity[0] < 0:
                    if movement["Blue"]["Running Shooting"]["Left"][1] == len(movement["Blue"]["Running Shooting"]["Left"][0]) - 1:
                        movement["Blue"]["Running Shooting"]["Left"][1] = -1
                    movement["Blue"]["Running Shooting"]["Left"][1] += 1
                    return movement["Blue"]["Running Shooting"]["Left"][0][movement["Blue"]]
            elif not shooting:

                # Idle Left
                if velocity[0] == 0 and velocity[1] == 0:
                    if movement["Blue"]["Standing"]["Left"][1] == len(movement["Blue"]["Standing"]["Left"][0]) - 1:
                        movement["Blue"]["Standing"]["Left"][1] = -1
                    movement["Blue"]["Standing"]["Left"][1] += 1
                    return movement["Blue"]["Standing"]["Left"][0][movement["Blue"]["Standing"]["Left"][1]],position
                
                # Walking Left
                if -15 <= velocity[0] < 0 and velocity[1] == 0:
                    if movement["Blue"]["Walking"]["Left"][1] == len(movement["Blue"]["Walking"]["Left"][0]) - 1:
                        movement["Blue"]["Walking"]["Left"][1] = -1
                    movement["Blue"]["Walking"]["Left"][1] += 1
                    return movement["Blue"]["Walking"]["Left"][0][movement["Blue"]["Walking"]["Left"][1]],position
                
                # Running Left
                if -30 <= velocity[0] < -15 and velocity[1] == 0:
                    return movement["Blue"]["Running"]["Left"],position
                
                # Jumping Left
                if velocity[1] > 0:
                    return movement["Blue"]["Jumping"]["Left"],position
                
                #Falling Left
                if velocity[1] < 0:
                    return movement["Blue"]["Falling"]["Left"],position
                
                if velocity[0] > 0:
                    return movement["Blue"]["Running"]["Left"],position

        elif orientation == "right":
            if shooting:
                
                # Idle Right WITH GUN
                if velocity[0] == 0 and velocity[1] == 0:
                    return movement["Blue"]["Idle Shooting"]["Right"],position
                
                # Walking Right WITH GUN
                if 0 < velocity[0] < 15 and velocity[1] == 0:
                    if movement["Blue"]["Walking Shooting"]["Right"][1] == len(movement["Blue"]["Walking Shooting"]["Right"][0]) - 1:
                        movement["Blue"]["Walking Shooting"]["Right"][1] = -1
                    movement["Blue"]["Walking Shooting"]["Right"][1] += 1
                    return movement["Blue"]["Walking Shooting"]["Right"][0][movement["Blue"]["Walking Shooting"]["Right"][1]],position
                
                # Running Right WITH GUN
                if 15 <= velocity[0] <= 30 and velocity[1] == 0:
                    if movement["Blue"]["Running Shooting"]["Right"][1] == len(movement["Blue"]["Running Shooting"]["Right"][0]) - 1:
                        movement["Blue"]["Running Shooting"]["Right"][1] = -1
                    movement["Blue"]["Running Shooting"]["Right"][1] += 1
                    return movement["Blue"]["Running Shooting"]["Right"][0][movement["Blue"]["Running Shooting"]["Right"][1]],position
                
                # Jumping Right WITH GUN
                if velocity[1] > 0:
                    return movement["Blue"]["Jumping Shooting"]["Right"],position
                
                # Falling Right WITH GUN
                if velocity[1] < 0:
                    return movement["Blue"]["Falling Shooting"]["Right"],position

                if velocity[0] < 0:
                  if movement["Blue"]["Running Shooting"]["Right"][1] == len(movement["Blue"]["Running Shooting"]["Right"][0]) - 1:
                        movement["Blue"]["Running Shooting"]["Right"][1] = -1
                  movement["Blue"]["Running Shooting"]["Right"][1] += 1
                  return movement["Blue"]["Running Shooting"]["Right"][0][movement["Blue"]["Running Shooting"]["Right"][1]],position
            elif not shooting:
                
                # Idle Right
                if velocity[0] == 0 and velocity[1] == 0:
                    if movement["Blue"]["Standing"]["Right"][1] == len(movement["Blue"]["Standing"]["Right"][0]) - 1:
                        movement["Blue"]["Standing"]["Right"][1] = -1
                    movement["Blue"]["Standing"]["Right"][1] += 1
                    return movement["Blue"]["Standing"]["Right"][0][movement["Blue"]["Standing"]["Right"][1]],position
                
                # Walking Right
                if 0 < velocity[0] < 15 and velocity[1] == 0:
                    if movement["Blue"]["Walking"]["Right"][1] == len(movement["Blue"]["Walking"]["Right"][0]) - 1:
                        movement["Blue"]["Walking"]["Right"][1] = -1
                    movement["Blue"]["Walking"]["Right"][1] += 1
                    return movement["Blue"]["Walking"]["Right"][0][movement["Blue"]["Walking"]["Right"][1]],position
                
                # Running Right
                if 15 <= velocity[0] <= 30 and velocity[1] == 0:
                    return movement["Blue"]["Running"]["Right"],position
                
                # Jumping Right
                if velocity[1] > 0:
                    return movement["Blue"]["Jumping"]["Right"],position
                
                #Falling Right
                if velocity[1] < 0:
                    return movement["Blue"]["Falling"]["Right"],position
                
                if velocity[0] < 0:
                    return movement["Blue"]["Running"]["Right"],position
    else:
        ##print('TEAM %s POSITION %s VELOCITY %s AERIAL %s SHOOTING %s ORIENTATION %s'%(team, position, velocity, aerial, shooting, orientation ) )
        if orientation == "left":
            if shooting:
                #Idle Left WITH GUN
                if velocity[0] == 0 and velocity[1] == 0:
                    return movement["Red"]["Idle Shooting"]["Left"],position
                
                # Walking Left WITH GUN
                if -15 <= velocity[0] < 0 and velocity[1] == 0:
                    if movement["Red"]["Walking Shooting"]["Left"][1] == len(movement["Red"]["Walking Shooting"]["Left"][0]) - 1:
                        movement["Red"]["Walking Shooting"]["Left"][1] = -1
                    movement["Red"]["Walking Shooting"]["Left"][1] += 1
                    return movement["Red"]["Walking Shooting"]["Left"][0][movement["Red"]["Walking Shooting"]["Left"][1]],position
                
                # Running Left WITH GUN
                if -30 <= velocity[0] < -15 and velocity[1] == 0:
                    if movement["Red"]["Running Shooting"]["Left"][1] == len(movement["Red"]["Running Shooting"]["Left"][0]) - 1:
                        movement["Red"]["Running Shooting"]["Left"][1] = -1
                    movement["Red"]["Running Shooting"]["Left"][1] += 1
                    return movement["Red"]["Running Shooting"]["Left"][0][movement["Red"]["Running Shooting"]["Left"][1]],position
                
                # Jumping Left WITH GUN
                if velocity[1] > 0:
                    return movement["Red"]["Jumping Shooting"]["Left"],position
                
                # Falling Left WITH GUN
                if velocity[1] < 0:
                    return movement["Red"]["Falling Shooting"]["Left"],position
                
                if velocity[0] > 0:
                    if movement["Red"]["Running Shooting"]["Left"][1] == len(movement["Red"]["Running Shooting"]["Left"][0]) - 1:
                        movement["Red"]["Running Shooting"]["Left"][1] = -1
                    movement["Red"]["Running Shooting"]["Left"][1] += 1
                    return movement["Red"]["Running Shooting"]["Left"][0][movement["Red"]["Running Shooting"]["Left"][1]],position
            elif not shooting:
                
                # Idle Left
                if velocity[0] == 0 and velocity[1] == 0:
                    if movement["Red"]["Standing"]["Left"][1] == len(movement["Red"]["Standing"]["Left"][0]) - 1:
                        movement["Red"]["Standing"]["Left"][1] = -1
                    movement["Red"]["Standing"]["Left"][1] += 1
                    return movement["Red"]["Standing"]["Left"][0][movement["Red"]["Standing"]["Left"][1]],position
                
                # Walking Left
                #print("gets here")
                if -15 <= velocity[0] < 0 and velocity[1] == 0:
                    if movement["Red"]["Walking"]["Left"][1] == len(movement["Red"]["Walking"]["Left"][0]) - 1:
                        movement["Red"]["Walking"]["Left"][1] = -1
                    movement["Red"]["Walking"]["Left"][1] += 1
                    return movement["Red"]["Walking"]["Left"][0][movement["Red"]["Walking"]["Left"][1]],position
                
                # Running Left
                if -30 <= velocity[0] < -15 and velocity[1] == 0:
                    return movement["Red"]["Running"]["Left"],position
                
                # Jumping Left
                if velocity[1] > 0:
                    return movement["Red"]["Jumping"]["Left"],position
                
                #Falling Left
                if velocity[1] < 0:
                    return movement["Red"]["Falling"]["Left"],position

                if velocity[0] > 0:
                  return movement['Red']["Running"]["Left"],position 
        elif orientation == "right":
            if shooting:
                
                # Idle Right WITH GUN
                if velocity[0] == 0 and velocity[1] == 0:
                    return movement["Red"]["Idle Shooting"]["Right"],position
                
                # Walking Right WITH GUN
                if 0 < velocity[0] < 15 and velocity[1] == 0:
                    if movement["Red"]["Walking Shooting"]["Right"][1] == len(movement["Red"]["Walking Shooting"]["Right"][0]) - 1:
                        movement["Red"]["Walking Shooting"]["Right"][1] = -1
                    movement["Red"]["Walking Shooting"]["Right"][1] += 1
                    return movement["Red"]["Walking Shooting"]["Right"][0][movement["Red"]["Walking Shooting"]["Right"][1]],position
                
                # Running Right WITH GUN
                if 15 <= velocity[0] <= 30 and velocity[1] == 0:
                    if movement["Red"]["Running Shooting"]["Right"][1] == len(movement["Red"]["Running Shooting"]["Right"][0]) - 1:
                        movement["Red"]["Running Shooting"]["Right"][1] = -1
                    movement["Red"]["Running Shooting"]["Right"][1] += 1
                    return movement["Red"]["Running Shooting"]["Right"][0][movement["Red"]["Running Shooting"]["Right"][1]],position
                
                # Jumping Right WITH GUN
                if velocity[1] > 0:
                    return movement["Red"]["Jumping Shooting"]["Right"],position
                
                # Falling Right WITH GUN
                if velocity[1] < 0:
                    return movement["Red"]["Falling Shooting"]["Right"],position
                  
                if velocity[0] < 0:
                  if movement["Red"]["Running Shooting"]["Right"][1] == len(movement["Red"]["Running Shooting"]["Right"][0]) - 1:
                      movement["Red"]["Running Shooting"]["Right"][1] = -1
                  movement["Red"]["Running Shooting"]["Right"][1] += 1
                  return movement["Red"]["Running Shooting"]["Right"][0][movement["Red"]["Running Shooting"]["Right"][1]],position

            elif not shooting:
                
                # Idle Right
                if velocity[0] == 0 and velocity[1] == 0:
                    if movement["Red"]["Standing"]["Right"][1] == len(movement["Red"]["Standing"]["Right"][0]) - 1:
                        movement["Red"]["Standing"]["Right"][1] = -1
                    movement["Red"]["Standing"]["Right"][1] += 1
                    return movement["Red"]["Standing"]["Right"][0][movement["Red"]["Standing"]["Right"][1]],position
                
                # Walking Right
                if 0 < velocity[0] < 15 and velocity[1] == 0:
                    if movement["Red"]["Walking"]["Right"][1] == len(movement["Red"]["Walking"]["Right"][0]) - 1:
                        movement["Red"]["Walking"]["Right"][1] = -1
                    movement["Red"]["Walking"]["Right"][1] += 1
                    return movement["Red"]["Walking"]["Right"][0][movement["Red"]["Walking"]["Right"][1]],position
                
                # Running Right
                if 15 <= velocity[0] <= 30 and velocity[1] == 0:
                    return movement["Red"]["Running"]["Right"],position
                
                # Jumping Right
                if orientation == "Right" and velocity[1] > 0:
                    return movement["Red"]["Jumping"]["Right"],position
                
                #Falling Right
                if velocity[1] < 0:
                    return movement["Red"]["Falling"]["Right"],position
                
                if velocity[0] < 0:
                  return movement['Red']['Running']['Right'],position
              

######################################################################
TIME_PER_FRAME = .1 #Assuming 10 FPS, doesn't have to be accurate just random estimate for speed weight
BULLET_WEIGHT = .6  #weight for bullet velocity to be added to persons velocity
PERSON_SHOOT_COOLDOWN = 60 #30 frames for bullet cooldown
PERSON_MAX_SPEED = 45
PERSON_HEIGHT = 70
PERSON_WIDTH  = 100
BULLET_RADIUS = 30
GRAVITY_STRENGTH = 10
SCREEN_WIDTH = pygame.display.Info().current_w
SCREEN_HEIGHT = pygame.display.Info().current_h
GAME_OVER_IMAGE = pygame.image.load("game_over.jpg")
BULLET_KNOCKBACK_VERTICAL = 10
JUMP_STRENGTH = 10
BULLET_SPEED = 120
BULLET_LIFETIME = 4
ACCELERATE_DOWN = 3
ENERGY_BALL_1 = pygame.image.load("energyball1.png")
ENERGY_BALL_2 = pygame.image.load("energyball2.png")

PERSON_IMAGES = list() #load this up with loaded images later, used by Displayable constructors
                       #to hold images
BULLET_IMAGES = list()

cv2.namedWindow('rect', cv2.WINDOW_NORMAL)
cv2.resizeWindow('rect', (SCREEN_WIDTH, SCREEN_HEIGHT) )

cv2.imshow('rect', cv2.rectangle(im.copy(), (minX, minY), (maxX, maxY), (255,0,0), thickness = 4 ) )
cv2.waitKey(0)

INTERACTIONS = defaultdict(list)

def cleanup_dead_bullets():
  current = time.time()

  indexarr = np.array(range(len(INTERACTIONS['Bullet'])) )
  for bnum in indexarr:
    if current - INTERACTIONS['Bullet'][bnum].birth > BULLET_LIFETIME:
      INTERACTIONS['Bullet'].pop(bnum)
      indexarr -= 1

def apply_gravity():
  for person in INTERACTIONS['Person']:
    if person.aerial:
      person.velocity[1] += GRAVITY_STRENGTH
  for bullet in INTERACTIONS['Bullet']:
    if bullet.touching_floor():
      bullet.velocity[1] *= -.8

class InteractBox:
  def __init__(self, x, y, w, h):
    self.topleft = np.array([x,y], dtype = np.float)
    self.w = w
    self.h = h

  def contains(self, position):
    if (position[1] < self.topleft[1] + self.h) and (position[1] > self.topleft[1]):
      if (position[0] > self.topleft[0]) and (position[0] < self.topleft[0] + self.w):
        return True
    return False

class InteractCircle:
  def __init__(self, x, y, r):
    self.center = np.array([x,y], dtype = np.float)
    self.r = r

class Background(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

class Moveable:
    def __init__(self, 
      startingpos = np.array([0,0], dtype = np.float), 
      startingvel = np.array([0,0], dtype = np.float)
      ):
        #print('Moveable')
        #All vectors are (x, y)
        self.position = np.array(startingpos, dtype = np.float)
        self.velocity = np.array(startingvel, dtype = np.float)
    
    def next_frame(self):
      if len(INTERACTIONS['Box']) == 0:
        self.position = self.position + self.velocity * TIME_PER_FRAME
        return

      while 1:
        nextpos = self.position + self.velocity * TIME_PER_FRAME
        for box in INTERACTIONS['Box']:
          if not box.contains(nextpos):
            self.position = nextpos
            return
        else:
          ##print('MULTING', self.team)
          self.velocity *= .75

class Person(Moveable):
    TEAM_NEXT = 0

    def __init__(self):
        self.aerial = False
        self.shoot_cooldown = False
        self.last_shot = 0
        self.get_team()
        Moveable.__init__(self, ( [SCREEN_WIDTH/3,40] if self.team == 0 else [SCREEN_WIDTH*2/3,40] ), [0,0] )
    
        INTERACTIONS['Person'].append(self)
        self.lives = 3

        self.hitbox = InteractBox(self.position[0], self.position[1], 
          w = PERSON_WIDTH, h = PERSON_HEIGHT)
        
        self.orientation = ('right' if self.team == 0 else 'left')
    
    def get_team(self):
        self.team = Person.TEAM_NEXT
        Person.TEAM_NEXT += 1
        if self.team > 1:
          raise RuntimeError('Change team descriptor from int to flag')
    
    def move(self, command):
      if command == 'shoot':
        Bullet(self.position, [BULLET_SPEED if self.orientation == 'right' else -BULLET_SPEED, 0], self.team)
      elif command == 'moveleft':
        self.velocity[0] = max(self.velocity[0] - 3, -PERSON_MAX_SPEED)
        self.orientation = 'left'
      elif command == 'moveright':
        self.velocity[0] = min(self.velocity[0] + 3, PERSON_MAX_SPEED)
        self.orientation = 'right'
      elif command == 'shoot':
        if self.shoot_cooldown == False:
          Bullet( 
                  self.position, 
                  [BULLET_SPEED if self.orientation == 'right' else -BULLET_SPEED, 0],
                  self.team )
          self.shoot_cooldown = True
          self.last_shot = PERSON_SHOOT_COOLDOWN
      elif command == 'acceleratedown':
        self.velocity[1] += ACCELERATE_DOWN
      elif command == 'jump':
        if not self.aerial:
            self.aerial = True
            self.velocity[1] -= JUMP_STRENGTH
      elif command == 'stop':
        if not self.aerial:
          self.velocity[0] = 0
      else:
        raise Exception('Passed in command [' + command + '] that isnt one of the accepted ones' )
    
    def next_frame(self):
      Moveable.next_frame(self)
      for box in INTERACTIONS['Box']:
        if self.position[1] == box.topleft[1]:
            if (self.position[0] < box.topleft[0]+box.w) and (self.position[0] > box.topleft[0]):
                self.aerial = False
                return
      else:
        self.aerial = True

class Bullet(Moveable):
    def __init__(self, startingpos, startingvel, team):
        Moveable.__init__(self, startingpos, startingvel)
        self.team = team
        self.birth = time.time()
        self.hitbox = InteractCircle(self.position[0], self.position[1], 
          r = BULLET_RADIUS)
        
        INTERACTIONS['Bullet'].append(self)
    
    def touching_floor(self):
      for box in INTERACTIONS['Box']:
        if self.position[1] == box.topleft[1]:
          if (self.position[0] < (box.topleft[0] + box.w)) and (self.position[0] > box.topleft[0]):
            return True
      return False

class Box:
    def __init__(self, topleft, w, h):
        INTERACTIONS['Box'].append(self)
        self.topleft = topleft
        self.w = w
        self.h = h

    def contains(self, position):
      if (position[1] < self.topleft[1] + self.h) and (position[1] > self.topleft[1]):
        if (position[0] > self.topleft[0]) and (position[0] < self.topleft[0] + self.w):
          return True
      return False


def get_commands(joysticks):
    '''
        Process inputs and turn them into commands.
        Return list of commands.
        List of commands is: jump, shoot, acceleratedown, moveleft,
            moveright, repeat for Person2
    '''
    player1moves = []
    player2moves = []
    for event in pygame.event.get():
        if event.type == pygame.JOYBUTTONDOWN:
            for joystick in joysticks:
                if joystick.get_button(1):
                    if joystick.get_id() == 0:
                        player1moves.append("shoot")
                    else:
                        player2moves.append("shoot")
                elif joystick.get_button(3):
                    if joystick.get_id() == 0:
                        player1moves.append("jump")
                    else:
                        player2moves.append("jump")
    for joystick in joysticks:  # Iterate over the available joysticks.
        numhats = joystick.get_numhats()
        if joystick.get_button(4): #if a player presses y, jump
            if joystick.get_id() == 0:
                player1moves.append("jump")
            else:
                player2moves.append("jump")
        for hat in range(numhats):  # Check all hats of the joystick.
            if joystick.get_id() == 0:
              if joystick.get_hat(hat) == (0,0):
                player1moves.append("stop")
              if joystick.get_hat(hat) == (0,1):
                player1moves.append("jump")
              if joystick.get_hat(hat) == (-1,0):
                player1moves.append("moveleft")
              if joystick.get_hat(hat) == (1,0):
                player1moves.append("moveright")
              if joystick.get_hat(hat) == (0,-1):
                player1moves.append("acceleratedown")
              if joystick.get_hat(hat) == (1,1):
                player1moves.append("jump")
                player1moves.append("moveright")
              if joystick.get_hat(hat) == (1,-1):
                player1moves.append("acceleratedown")
                player1moves.append("moveright")
              if joystick.get_hat(hat) == (-1,1):
                player1moves.append("jump")
                player1moves.append("moveleft")
              if joystick.get_hat(hat) == (-1,-1):
                player1moves.append("acceleratedown")
                player1moves.append("moveleft")
            else:
              if joystick.get_hat(hat) == (0,0):
                player2moves.append("stop")
              if joystick.get_hat(hat) == (0,1):
                player2moves.append("jump")
              if joystick.get_hat(hat) == (-1,0):
                player2moves.append("moveleft")
              if joystick.get_hat(hat) == (1,0):
                player2moves.append("moveright")
              if joystick.get_hat(hat) == (0,-1):
                player2moves.append("acceleratedown")
              if joystick.get_hat(hat) == (1,1):
                player2moves.append("jump")
                player2moves.append("moveright")
              if joystick.get_hat(hat) == (1,-1):
                player2moves.append("acceleratedown")
                player2moves.append("moveright")
              if joystick.get_hat(hat) == (-1,1):
                player2moves.append("jump")
                player2moves.append("moveleft")
              if joystick.get_hat(hat) == (-1,-1):
                player2moves.append("acceleratedown")
                player2moves.append("moveleft")

    return player1moves, player2moves
''' 
    Initalize Screens/Backgrounds
'''
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT),pygame.FULLSCREEN)
BackGround = Background('background.jpg', [0,0])

#GameOver = Displayable(GAME_OVER_IMAGE,GAME_OVER_IMAGE.get_rect())
'''
    Initialize controllers
'''
joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
for joystick in joysticks:
    joystick.init()

'''
    Restart Game Loop 
'''

'''
    Main game loop
'''
filtered_squarelist = [((cor[0][0] * (SCREEN_WIDTH / 640), cor[0][1] * (SCREEN_HEIGHT / 480)), (cor[1][0] * (SCREEN_WIDTH / 640), cor[1][1] * (SCREEN_HEIGHT / 480))) for cor in filtered_squarelist]
screen.fill([255, 255, 255])

for ( minX, minY ), ( maxX, maxY ) in filtered_squarelist:
  Box( np.array([minX, minY]), maxX-minX, maxY-minY )
  print('Box made at position ', minX, minY, 'with w and h', maxX - minX, maxY - minY)

PERSON1 = Person()
PERSON2 = Person()

GAME_OVER = False
while not GAME_OVER:
    screen.fill([255, 255, 255])
    screen.blit(BackGround.image, BackGround.rect)

    '''
        check bullet -> person
        commands->person
        check person -> floor
        check bullet -> floor
        check person -> person

        increment position with step, each step is .1 second (can change)
    '''
    for interaction in INTERACTIONS["Person"]:
      try:
        img, pos = spriteEquivalent(interaction.team, interaction.position,interaction.velocity, interaction.aerial,interaction.shoot_cooldown, interaction.orientation)
        img = pygame.transform.scale(img,(PERSON_WIDTH,PERSON_HEIGHT))
        screen.blit(img, [pos[0], pos[1]])
      except TypeError:
        print("aint nobody got time for that")

    for interaction in INTERACTIONS["Bullet"]:
      if interaction.team == 0:
        screen.blit(pygame.transform.scale(ENERGY_BALL_1, (BULLET_RADIUS * 2, BULLET_RADIUS * 2)), [interaction.position[0] + BULLET_RADIUS, interaction.position[1] + BULLET_RADIUS])
      else:
        screen.blit(pygame.transform.scale(ENERGY_BALL_2, (BULLET_RADIUS * 2, BULLET_RADIUS * 2)), [interaction.position[0] + BULLET_RADIUS, interaction.position[1] + BULLET_RADIUS])
      


    bullet_index_arr = np.arange( len(INTERACTIONS['Bullet']) )
    #print('bullet_index_arr is ', bullet_index_arr)
    for person in INTERACTIONS['Person']:
      for bnum in bullet_index_arr:
        if len(INTERACTIONS['Bullet']) == 0:
          break
        #print(bullet_index_arr)
        #print(bnum)
        #print(len(INTERACTIONS['Bullet']) )
        if person.team != INTERACTIONS['Bullet'][bnum].team:
          if person.hitbox.contains( INTERACTIONS['Bullet'][bnum].position ):
            person.velocity += INTERACTIONS['Bullet'][bnum].velocity * BULLET_WEIGHT
            person.velocity[1] -= BULLET_KNOCKBACK_VERTICAL
            person.aerial = True

            INTERACTIONS['Bullet'].pop(bnum)
            bullet_index_arr -= 1
            #print('REMOVED BULLET')
        
    person1 = INTERACTIONS['Person'][0] #Do Person-Person calculations with Person 0 since only 2 people
    corners1 = [person1.hitbox.topleft, #topleft
                person1.hitbox.topleft + [0, person1.hitbox.h], #bottomleft
                person1.hitbox.topleft + [person1.hitbox.w, person1.hitbox.h], #bottomright
                person1.hitbox.topleft + [person1.hitbox.w, 0] ] #topright
    
    for corner in corners1:
      if INTERACTIONS['Person'][1].hitbox.contains( corner ):
        person1 = INTERACTIONS['Person'][0]
        person2 = INTERACTIONS['Person'][1]
        total = abs(person1.velocity) + abs(person2.velocity)
        person1.velocity = total/2
        person2.velocity = total/2
        person1.velocity[1] = -5
        person2.velocity[1] = -5
        person1.aerial = True
        person2.aerial = True
        break

    commands = get_commands(joysticks)
    #print(commands[0],commands[1])


    #Putting commands into people, discard commands now
    for person_number in range(2):
      for command in commands[person_number]:
        if command == 'moveleft':
          INTERACTIONS['Person'][person_number].move('moveleft')
        elif command == 'moveright':
          INTERACTIONS['Person'][person_number].move('moveright')
        elif command == 'jump':
          INTERACTIONS['Person'][person_number].move('jump')
        elif command == 'acceleratedown':
          INTERACTIONS['Person'][person_number].move('acceleratedown')
        elif command == 'shoot':
          INTERACTIONS['Person'][person_number].move('shoot')
      #print('----------------------------------------')
      #print(INTERACTIONS['Person'][person_number].team, person_number)

    if PERSON1.shoot_cooldown:
      PERSON1.last_shot -= 1
      if PERSON1.last_shot == 0:
        PERSON1.shoot_cooldown = False

    if PERSON2.shoot_cooldown:
      PERSON2.last_shot -= 1
      if PERSON2.last_shot == 0:
        PERSON2.shoot_cooldown = False
    
    for person in INTERACTIONS['Person']:
      person.next_frame()
    for bullet in INTERACTIONS['Bullet']:
      bullet.next_frame()

    for square in filtered_squarelist:
      pygame.draw.rect(screen, 1, pygame.Rect(square[0][0], square[0][1], square[1][0] - square[0][0], square[1][1] - square[0][1]))
      

    cleanup_dead_bullets()
    apply_gravity()
    pygame.display.flip()
'''
End Game Loop
'''
# while True:
#     GameOver.render(screen)
#     commands = get_commands(joysticks)
#     for player_commands in commands:
#         for indiv_commands in player_commands:
#             if indiv_commands == "shoot":
#                 break
    
