﻿Перед тем, как запустить скрипт, требуется установить необходимое для работы ПО (при его отсутствии) - папка Software

Способы установки:

1) автоматический, с помощью запуска setup.bat
2) пошаговый, "вручную" (см. далее)

#############################################################################################################################################################

Необходимое ПО:

1) python 3.5
2) pip
3) numpy

#############################################################################################################################################################

Порядок установки:

1) Установка python 3.5

	а) Запустить файл python-3.5.0.exe
	б) Выбрать пункт 'Customize installation'
	в) Проверить наличие галочек у пунктов 'pip', 'py launcher' и 'for all users'
	г) Next
	д) Проверить наличие галочек у пунктов 'install for all users', 'associate files with Python' и 'Add Python to environment variables'
	е) Исправить путь для установки (Customize install location: C:\Python35\)
	ж) Install

2) Установка (обновление) pip

	а) Запустить файл get-pip.py
	б) Если этот файл не удается открыть, в списке предлагаемых программ, указать Python Launcher (или в директории C:\Python35 выбрать файл python.exe)

3) Установка numpy (запуск install_numpy.bat)

############################################################################################################################################################

Запуск программы:

1) запуск run_script.bat
2) а) запустить cmd
   б) перейти в папку со скриптом
   в) ввести команду python MultiPieScript.py