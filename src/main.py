from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import Recognitor
import io
import cv2


with open('token.txt', 'r') as f:
    token = f.readline()

updater = Updater(token=token)
dispatcher = updater.dispatcher


# Обработка команд
def start_command(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Привет, давай пообщаемся?')


def text_message(bot, update):
    response = 'Получил Ваше сообщение: ' + update.message.text
    bot.send_message(chat_id=update.message.chat_id, text=response)


def get_closest(photos, desired_size):
    def diff(p):
        return p.width - desired_size[0], p.height - desired_size[1]

    def norm(t):
        return abs(t[0] + t[1] * 1j)

    return min(photos, key=lambda p: norm(diff(p)))


def image_message(bot, update):
    photo = update.message.photo[-1].file_id
    photo = bot.get_file(photo)
    photo.download('{}.jpg'.format(update.message.photo[-1].file_id))
    result_photo = Recognitor.recognize('{}.jpg'.format(update.message.photo[-1].file_id))
    cv2.imwrite("result.png", result_photo)
    bot.send_photo(chat_id=update.message.chat_id, photo=open('result.png', 'rb'))


# Хендлеры
start_command_handler = CommandHandler('start', start_command)
text_message_handler = MessageHandler(Filters.text, text_message)
photo_message_handler = MessageHandler(Filters.photo, image_message)
# Добавляем хендлеры в диспетчер
dispatcher.add_handler(start_command_handler)
dispatcher.add_handler(text_message_handler)
dispatcher.add_handler(photo_message_handler)
# Начинаем поиск обновлений
updater.start_polling(clean=True)
# Останавливаем бота, если были нажаты Ctrl + C
updater.idle()
