import random
import sys
from collections.abc import Iterable
from functools import reduce
import pygame
import math
import numpy as np
from RL_brain import RL

class game_2048():
    def __init__(self):
        self.score = 0
        self.highest_block = 0
        self.width, self.height = (370, 600)
        self.margin_size = 10
        self.block_size = 80
        self.show_screen = 0
        self.max_illegal_move = 100     # max number of illegal actions
        self.num_illegal_move = 0
        self.increased_epsilon_countdown = 0

    def new_game(self):
        mat = np.zeros((4, 4))
        mat = self.add_random_block(mat)
        mat = self.add_random_block(mat)
        self.score = 0
        self.highest_block = 0
        self.num_illegal_move = 0
        return mat

    def add_random_block(self, mat):
        i = random.randint(0, len(mat) - 1)
        j = random.randint(0, len(mat) - 1)
        while(mat[i][j] != 0):
            i = random.randint(0, len(mat) - 1)
            j = random.randint(0, len(mat) - 1)
        mat[i][j] = random.randint(1, 2) * 2
        return mat

    def print_mat(self, mat):
        for row in mat:
            print(row)

    def change_left_right(self, mat):
        new_mat = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                new_mat[i][j] = mat[i][3-j]
        return new_mat

    def change_row_column(self, mat):
        new_mat = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                new_mat[i][j] = mat[j][i]
        return new_mat

    def compress(self, mat):
        move_or_not = 0
        new_mat = np.zeros((4, 4))
        for i in range(4):
            index = 0
            for j in range(4):
                if mat[i][j] != 0:
                    if index != j:
                        move_or_not = 1
                    new_mat[i][index] = mat[i][j]
                    index += 1
        return move_or_not, new_mat

    def merge(self, mat):
        move_or_not = 0
        for i in range(4):
            for j in range(3):
                if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                    move_or_not = 1
                    mat[i][j] *= 2
                    self.score += mat[i][j]
                    self.highest_block = max(self.highest_block, mat[i][j])
                    mat[i][j+1] = 0
        return move_or_not, mat

    def move_left(self, mat):
        move_or_not1, mat = self.compress(mat)
        move_or_not2, mat = self.merge(mat)
        move_or_not3, mat = self.compress(mat)
        move_or_not = move_or_not1 or move_or_not2 or move_or_not3 
        return move_or_not, mat

    def move_right(self, mat):
        mat = self.change_left_right(mat)
        move_or_not, mat = self.move_left(mat)
        mat = self.change_left_right(mat)
        return move_or_not, mat

    def move_up(self, mat):
        mat = self.change_row_column(mat)
        move_or_not, mat = self.move_left(mat)
        mat = self.change_row_column(mat)
        return move_or_not, mat

    def move_down(self, mat):
        mat = self.change_row_column(mat)
        move_or_not, mat = self.move_right(mat)
        mat = self.change_row_column(mat)
        return move_or_not, mat

    def gameover_or_not(self, mat):
        flag = 0
        for i in range(4):
            for j in range(4):
                if mat[i][j] == 0:
                    return 0
                elif j + 1 < 4 and mat[i][j] == mat[i][j+1]:
                    return 0
                elif i + 1 < 4 and mat[i][j] == mat[i + 1][j]:
                    return 0
        return 1

    def draw_back(self, screen):
        for i in range(4):
            for j in range (4):
                x = self.margin_size * (j + 1) + self.block_size * j
                y = self.margin_size * (i + 1) + self.block_size * i + self.height - self.width
                pygame.draw.rect(screen, pygame.Color('#f9f6f2'), (x, y, self.block_size,self.block_size))

    color_map = {
        0: "#cdc1b4",
        2: "#eee4da",
        4: "#ede0c8",
        8: "#f2b179",
        16: "#f59563",
        32: "#f67c5f",
        64: "#f65e3b",
        128: "#edcf72",
        256: "#edcc61",
        512: "#edc850",
        1024: "#edc53f",
        2048: "#edc22e",
    }


    def draw_num(self, screen, mat):
        font_size = self.block_size - 15
        font = pygame.font.Font(None, font_size)
        for i in range(4):
            for j in range(4):
                if mat[i][j] != 0:
                    x = self.margin_size * (j + 1) + self.block_size * j 
                    y = self.margin_size * (i + 1) + self.block_size * i + self.height - self.width
                    font_color = pygame.Color('#776e65')
                    text = font.render(str(int(mat[i][j])), True, font_color)
                    rect = text.get_rect()
                    rect.centerx, rect.centery = x + self.block_size / 2, y + self.block_size / 2
                    pygame.draw.rect(screen, pygame.Color(self.color_map.get(mat[i][j], '#776e65')), (x, y, self.block_size, self.block_size))
                    screen.blit(text, rect)

    def draw_score(self, screen, score):
        font_size = self.block_size - 15
        font = pygame.font.Font(None, font_size)

        font_color = pygame.Color('#f9f6f2')
        text = font.render('score:'+str(int(score)), True, font_color)
        screen.blit(text, (25, 25))

    def draw_gameover(self, screen):
        font_size = self.block_size
        font = pygame.font.Font(None, font_size)

        font_color = pygame.Color('#f9f6f2')
        text = font.render('Gameover.', True, font_color)
        screen.blit(text, (25, 200))

        font_size = self.block_size -30
        font = pygame.font.Font(None, font_size)
        text = font.render('Press R to restart.', True, font_color)
        screen.blit(text, (25, 270))

    def draw_moves_episode_epsilon(self, screen, moves, episode, epsilon):
        font_size = self.block_size
        font = pygame.font.Font(None, font_size)
        font_color = pygame.Color('#f9f6f2')

        text = font.render('moves:'+str(moves), True, font_color)
        screen.blit(text, (25, 85))

        text = font.render('episode:'+str(episode), True, font_color)
        screen.blit(text, (25, 150))

        text = font.render('epsilon'+str(epsilon), True, font_color)
        screen.blit(text, (25, 200))

    def space_count(self, mat1):
        num = 0
        for i in range(4):
            for j in range(4):
                if mat1[i][j] == 0:
                    num+=1
        return num

    def play_2048(self, screen, state ,action, moves, episode, epsilon):
        previous_score = self.score
        clock = pygame.time.Clock()
        gameover = 0
        pre_space_num = self.space_count(state)
        if action == 0:
            #print('up')
            move_or_not, state = self.move_up(state)
        elif action == 1:
            #print('down')
            move_or_not, state = self.move_down(state)
        elif action == 2:
            #print('left')
            move_or_not, state = self.move_left(state)
        elif action == 3:
            #print('right')
            move_or_not, state = self.move_right(state)

        gameover = self.gameover_or_not(state)

        reward = 0
        if move_or_not == 1:
            new_space_num = self.space_count(state)
            #reward += (new_space_num - pre_space_num)
            reward = self.score - previous_score
            state = self.add_random_block(state)
            reward += 1
        else:
            self.num_illegal_move += 1
            #reward += self.num_illegal_move * (-2)
            reward += (-5)
            if self.num_illegal_move > self.max_illegal_move:
                gameover = 1

            #if self.num_illegal_move > self.max_illegal_move and self.increased_epsilon_countdown == 0: 
                # when AI stuck in a state, which will let AI keep doing illegal moves.
                # we increse espilon temporarily
                #RL.change_epsilon(RL.init_epsilon)
                #self.increased_epsilon_countdown = self.max_illegal_move
        #if self.increased_epsilon_countdown > 0:
        #    self.increased_epsilon_countdown -= 1
        #    if self.increased_epsilon_countdown == 0:
        #        RL.epsilon_decay(episode, RL.total_episode)

        if gameover == 1:
            #print("gameover")
            #print(self.score)
            reward = -25


        
        if self.show_screen == 1:
            screen.fill(pygame.Color('#92877d'))

            self.draw_back(screen)
                
            self.draw_num(screen, state)
                
            self.draw_score(screen, self.score)

            self.draw_moves_episode_epsilon(screen, moves, episode, epsilon)

            if gameover:
                self.draw_gameover(screen)
                pygame.display.flip()
                pygame.time.delay(300)


        if self.show_screen == 1:
            pygame.display.flip()
            clock = pygame.time.Clock()
            clock.tick(15)
        

        #print('reward:'+str(reward))
        return state, reward, gameover

environment = game_2048()