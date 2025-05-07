import time

def count_yawns(states_list):
    """Count yawns in the states list based on transitions to state 1."""
    count = 0
    for i in range(1, len(states_list)):
        if states_list[i] == 1 and states_list[i - 1] != 1:
            count += 1
    return count

def detect_microsleep(states_list):
    """Check if there are 3 consecutive 0s (closed eyes) in the history."""
    sequence = ''.join(map(str, states_list))
    alert = '000' in sequence
    return alert

# ---------- Melodía de Mario Bros para alerta ----------

notes = {
    'E5': 659, 'C5': 523, 'G5': 784, 'G4': 392,
    'E4': 330, 'A4': 440, 'B4': 494, 'AS4': 466,
    'F5': 698, 'D5': 587, 'A5': 880
}

melody = [
    ('E5', 0.125), ('E5', 0.125), ('E5', 0.25),
    ('C5', 0.125), ('E5', 0.25), ('G5', 0.5),
    ('G4', 0.5),
    ('C5', 0.25), ('G4', 0.25), ('E4', 0.25),
    ('A4', 0.125), ('B4', 0.125), ('AS4', 0.125), ('A4', 0.25),
    ('G4', 0.125), ('E5', 0.125), ('G5', 0.125), ('A5', 0.25),
    ('F5', 0.125), ('G5', 0.25),
    ('E5', 0.125), ('C5', 0.125), ('D5', 0.125), ('B4', 0.25)
]

def play_mario_theme(pwm):

    """
    Plays the Super Mario Bros theme using a passive buzzer and PWM.

    Args:pwm (RPi.GPIO.PWM): The PWM object configured for the buzzer output.
    
    This function iterates over a predefined melody, where each note is played 
    by setting the buzzer's frequency and duty cycle. After each note, a brief 
    pause is added to separate the tones.
    """

    for note, duration in melody:
        if note in notes:
            freq = notes[note]
            pwm.ChangeFrequency(freq)
            pwm.start(80)
            time.sleep(duration)  
            pwm.stop()
            time.sleep(duration * 0.1)
