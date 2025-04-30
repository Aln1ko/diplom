#Визначення швидкості рухомої платформи за допомогою оптичних датчиків

В проекті реалізовані нейронні мережі для вирішення задачі визначення швидкості рухомої платформи.

#Функції
- Кастомна та стандартні архітектури CNN
- Підтримка навчання, валідації та тестування
- Доповнення та попередня обробка даних
- Візуалізація метрик навчання (втрати, точність)
- Контрольні точки та завантаження моделі
- Звіт про метрику оцінки

#Структура проекту
diplom
|-additional
|-classic_methods
|-models

#Daset
в проекті використовується KITTI Odometry Dataset, його можливо отримати за посиланням
https://www.cvlibs.net/datasets/kitti/eval_odometry.php 
або на гугл диску(але тут тільки по окремим папкам, для яких дані для навчання)
00 - https://drive.google.com/drive/u/0/folders/1KNLwCAPfbbAXG2dllrzVJu4ONx-5kXZ_
01 - https://drive.google.com/drive/u/0/folders/17OKE0Hm15zwhg1RqZIlIX_i5x-w44wxO
02 - https://drive.google.com/drive/u/0/folders/19-c26ylk5Gr2f8twuVdlj6S1sqxtNWIG
03 - https://drive.google.com/drive/u/0/folders/19AinnGOY3L-ZNawdGNjJLWITrHAL5s_v
04 - https://drive.google.com/drive/u/0/folders/1ThvgaN8dqU1Or7nh8q3syp34zxRUrDa7
05 - https://drive.google.com/drive/u/0/folders/11CcV4tRfgvuNmZtwtpXD4V8ZUiDqMXQg
06 - https://drive.google.com/drive/u/0/folders/1GqEh81JPvHTO4tjfeRG0Wq9BRxcpd-7h
07 - https://drive.google.com/drive/u/0/folders/1dko99PeSfQHHp6UHmyYx7PGgYKKqBcK8
08 - https://drive.google.com/drive/u/0/folders/1j6voTkEcjVClUgwSQzmdTaZTavx6uj1r
09 - https://drive.google.com/drive/u/0/folders/1H39AP_sTMufHKUuVzAJplCMteGPwnwfU
10 - https://drive.google.com/drive/u/0/folders/102buzjBlWYeSIpdj0dGWXctfV6HdQldX

В папці additional знаходяться програми для роботи з відео, по вирізання по довжинні відео, вирізання області інтересу, перетворення в послідовність картинок, та накладання яуихось данних на відео.
В папці classic_methods знаходяться классичні методи для вирішення задачі.
В папці models знаходяться моделі разом з завантаденням і претворенням данних, результатами. Дані в моделі завантажуються з гугл диска.