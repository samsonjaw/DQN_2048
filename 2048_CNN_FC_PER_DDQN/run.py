import tensorflow as tf
import pygame
from tb import logger
from RL_brain import RL
from environment_2048 import environment


width, height = (370, 600)
 
def run_2048():
    moves = 0
    
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((width, height))
    total_episode = RL.total_episode

    for episode in range(total_episode):

        RL.epsilon_decay(episode,total_episode)
        
        if episode % 10 == 0:
            RL.save_model()
        state = environment.new_game()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            #print(RL.epsilon)
            
            action = RL.choose_action(state)
            next_state, reward, gameover = environment.play_2048(screen, state, action, moves, episode, RL.epsilon)

            RL.store(state, action, reward, next_state, gameover)

            if (moves > 15) and (moves % 5 == 0):
                RL.replay(batch_size = 256)

            if moves % RL.target_update_freq == 0:
                RL.update_target()

            state = next_state
            if gameover:
                break
            moves += 1
        #print(environment.score)
        logger.log_scalar("game_score", environment.score, step = episode)
        logger.log_scalar("highest_block", environment.highest_block, step = episode)
    #print('over')
    

def main():
    run_2048()
    #print('run_over')

if __name__ == "__main__":
    main()