"""
Contains the objects that define the music theory of the model.
"""
from src.music_theory_classes import Qualities, Mode

qualities = Qualities(('M','major', {'C':0.4, 'E':0.3, 'G':0.3}),
                      ('m','Minor', {'C':0.4, 'E-':0.3, 'G':0.3}),
                      ('o','diminished', {'C':0.38, 'E-':0.3, 'G-':0.32}),
                      ('+','augmented', {'C':0.37, 'E':0.3, 'G#':0.33}),
                      ('maj7','major seventh', {'C':0.35, 'E':0.2, 'G':0.2, 'B':0.25}),
                      ('m7','minor seventh', {'C':0.35, 'E-':0.2, 'G':0.2, 'B-':0.25}),
                      ('7','dominant seventh', {'C':0.35, 'E':0.2, 'G':0.2, 'B-':0.25}),
                      ('o7','diminshed seventh', {'C':0.35, 'E-':0.1, 'G-':0.3, 'B--':0.25}),
                      ('o/7','half-diminished seventh', {'C':0.35, 'E-':0.1, 'G-':0.3, 'B-':0.25}),
                      ('It','italian augmented sixth', {'C':0.4,'E--':0.4,'G-':0.2}),
                      ('Fr','french augmented sixth', {'C':0.1,'E':0.4,'G-':0.4,'B-':0.1}),
                      ('Ger','german augmented sixth', {'C':0.4,'E--':0.4,'G-':0.1,'B--':0.1}))

major_mode = Mode('M',
               [('I',0,0,'M',1),
                ('I',0,0,'maj7',0.5), # ?
                ('II',1,2,'m',0.99),
                ('II',1,2,'m7',0.99),
                ('III',2,4,'m',0.8),
                ('III',2,4,'m7',0.5), # ?
                ('IV',3,5,'M',0.99),
                ('IV',3,5,'maj7',0.8),
                ('V',4,7,'M',0.99),
                ('V',4,7,'7',0.99),
                ('VI',5,9,'m',0.99),
                ('VI',5,9,'m7',0.8),
                ('VII',6,11,'o',0.99),
                ('VII',6,11,'o/7',0.99)])

minor_mode = Mode('m',
               [('I',0,0,'m',1),
                ('I',0,0,'m7',0.8),
                ('N',1,1,'M',0.95), #Napolitan
                ('II',1,2,'o',0.99),
                ('II',1,2,'o/7',0.99),
                ('III',2,3,'+',0.9),
                ('III',2,3,'M',0.6),
                ('III',2,3,'maj7',0.4), # ?
                ('IV',3,5,'m',0.99),
                ('IV',3,5,'m7',0.8),
                ('V',4,7,'M',0.99),
                ('V',4,7,'7',0.99),
                ('V',4,7,'m',0.8),
                ('VI',5,8,'M',0.99),
                ('VI',5,8,'maj7',0.6), # ?
                ('VII',6,11,'o',0.99),
                ('VII',6,11,'o7',0.99),
                ('VII',6,10,'M',0.8),
                ('It',3,6,'It',0.9),
                ('Ger',3,6,'Ger',0.9),
                ('Fr',1,2,'Fr',0.9)])
