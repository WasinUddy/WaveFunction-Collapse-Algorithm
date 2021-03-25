
# * Author : Wasin Silakong
# * Copyright (c) 2021 Careless Dev Squad All rights reserved.



# * Import required Library

# * Import Numpy to create 2D array
import numpy as np

# * Import random to simulate True Random 
import random 

# * Import opencv for image work
import cv2

# * Import Tqdm for progress bar
from tqdm import tqdm

# * Import Imageio for creating Animation
import imageio

# * Creating Class

# * WFC2D class to make Wavefunction collapse model
class WFC2D:

    # * class constructor
    def __init__(self, generateimage_size: tuple, combinations: dict, initial_probability: dict):

        # * Get Image Size
        self.image_height, self.image_width = generateimage_size

        # Tile Dim
        self.tiledim = cv2.imread('tileset/0.png').shape

        # * Collapse Rules
        self.combinations_dict = combinations

        # * Starting Probability
        self.initial_probability = initial_probability

        # * Create Superpositon Image called image
        self.super_image = self.__create_super_image()

        # * Create Array of superpostion Entropy
        self.__get_entropy = np.vectorize(lambda x: x.get_entropy())

        # * Create place holder for actual image aka generated Image
        self.image = np.full((self.image_height, self.image_width), None)

        # * collected Image for animation
        self.frame = []
        self.export_animation = False


# * ---------------------------------------Public Method-------------------------------------------

    # * RUN Until all wave has been collaps
    def run(self, export_animation=False):
        self.export_animation = export_animation
        while True:
            
            try:
                self.__run()
                
            except:
                # * Create Superpositon Image called image
                self.super_image = self.__create_super_image()

                # * Create Array of superpostion Entropy
                self.__get_entropy = np.vectorize(lambda x: x.get_entropy())

                # * Create place holder for actual image aka generated Image
                self.image = np.full((self.image_height, self.image_width), None)

                self.frame = []
            else:
                break

        return self.image

    # * Generate Collapsed Image
    def generate_image(self, file="Generated.png"):
        image_height, image_width = self.image.shape

        height = []
        for h in range(image_height):
            width = []
            for w in range(image_width):
                width.append(cv2.imread(f"tileset/{self.image[h][w]}.png"))
            height.append(np.hstack(width))

        img = np.vstack(height)
        
        cv2.imwrite(file, img)

    # * Generate Animation
    def generate_animation(self):
        if self.export_animation:
            image_height, image_width = self.image.shape
            print("Creating Animation")
            
            for count, frame in tqdm(enumerate(self.frame)):

                height = []
                for h in range(image_height):
                    width = []
                    for w in range(image_width):
                        if frame[h][w] is not None:
                            width.append(cv2.imread(f"tileset/{frame[h][w]}.png"))
                        else:
                            width.append(np.zeros(self.tiledim))
                    height.append(np.hstack(width))
                img = np.vstack(height)
                
                cv2.imwrite(f"animation/frames/{count}.png", img)
            
            files = [imageio.imread(f"animation/frames/{i}.png") for i in range(len(self.frame))]
            imageio.mimwrite("animation/animation.gif", files, fps=4)
                
        else:
            print("Animation Mode not chosen cannot create animation")

# * ---------------------------------------------------------------------------------------------


    # * finding minimum entropy    
    def __min_entropy(self):
        index_y, index_x = np.where(self.entropy_arr == self.entropy_arr.min())
        
        index_y = index_y.tolist()
        index_x = index_x.tolist()

        # * formatting data
        position = []
        for index, y in enumerate(index_y):
            position.append((y, index_x[index]))

        # * return random position that have minimum entropy    
    
        return self.__random_method(position)

    
    # * Collapse once
    def __collapse(self):

        # * Perform Quantum Judgement
        
        min_entropy_y, min_entropy_x = self.__min_entropy()
        
        self.super_image[min_entropy_y][min_entropy_x].judgement()
        self.image[min_entropy_y][min_entropy_x] = self.super_image[min_entropy_y][min_entropy_x].normal_position

        # * Changing Probability of others bad code copy from previous version LOL
        h = min_entropy_y
        w = min_entropy_x

        image_height = self.image_height
        image_width = self.image_width

        others_prob = self.combinations_dict[self.super_image[h][w].normal_position]

        # North
        if h + 1 <= image_height - 1:
            self.super_image[h+1][w].probability = self.__converter(set(self.super_image[h+1][w].probability.keys()) & set(self.__converter(others_prob["S"]).keys()))

        # South
        if h - 1 >= 0:
            self.super_image[h-1][w].probability = self.__converter(set(self.super_image[h-1][w].probability.keys()) & set(self.__converter(others_prob["N"]).keys()))
            
        # East
        if w + 1 <= image_width - 1:
            self.super_image[h][w+1].probability = self.__converter(set(self.super_image[h][w+1].probability.keys()) & set(self.__converter(others_prob["E"]).keys()))

        # West
        if w - 1 >= 0:
            self.super_image[h][w-1].probability = self.__converter(set(self.super_image[h][w-1].probability.keys()) & set(self.__converter(others_prob["W"]).keys()))


    # * run regardless of failing to collapse 
    def __run(self):
            self.__create_entropy_arr()
            while not self.__complete_check():
                self.__collapse()
                self.__create_entropy_arr()

                if self.export_animation:
                    fh= []
                    for h in range(self.image_height):
                        fw = []
                        for w in range(self.image_width):
                            fw.append(self.image[h][w])
                        fh.append(fw)

                    self.frame.append(np.array(fh))
    # * check for completation
    def __complete_check(self):
        return np.max(self.entropy_arr) == float('inf') and np.min(self.entropy_arr) == float('inf')


    # Create array containing entropy of all superposition
    def __create_entropy_arr(self):        
        self.entropy_arr = np.full((self.image_height, self.image_width), -1.0)
        for h in range(self.image_height):
            for w in range(self.image_width):
                self.entropy_arr[h][w] = self.super_image[h][w].get_entropy()
        return self.entropy_arr


    # Create image array that all position is superposition
    def __create_super_image(self):
        image = []
        for h in range(self.image_height):
            ws = []
            for w in range(self.image_width):
                ws.append(Superposition(self.initial_probability))
            image.append(ws)
        return np.array(image)


    # * Method use for random
    def __random_method(self, object):
        return random.choice(object)


    # * Just a simple function for convert data type
    def __converter(self, prob):
        p = {}
        for i in prob:
            p[i] = 1
        return p

    
# * Create Superposition class:
class Superposition:

    # * Class Constructor
    def __init__(self, init_probability: dict):

        # * Place Holder for judged Superposition object
        self.normal_position = None

        # * Place Holder for superposition object list
        self.super_position = None 

        # * Variable storing current avaliable choice and probability within
        self.probability = init_probability
        

    # * Making Decision
    def judgement(self):

        # *  Generate Probability list
        all_keys = self.probability.keys()
        random_list = []
        
        for key in all_keys:
            for i in range(self.probability[key]):
                random_list.append(key)

        # * Making Decision
        self.normal_position = self.__random_method(random_list)
        

    # * create_superposition list
    def create_superposition(self, keep=False):
        superposition = self.probability.keys()
        if keep is True:
            self.super_position = superposition
        return superposition
        

    # * Remove superposition list clear RAM
    def remove_superposition(self):
        self.super_position = None


    # Get Current Entropy of superposition if the judgement occure entropy -> infinity
    def get_entropy(self):
        if self.normal_position is None:
            return len(self.probability.keys())
        else:
            return float('inf')
        

    # * Method use for random
    def __random_method(self, object):
        return random.choice(object)
