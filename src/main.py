from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import Recognitor
import cv2
from telegram import ChatAction
from functools import wraps


def send_action(action):
    """Sends `action` while processing func command."""

    def decorator(func):
        @wraps(func)
        def command_func(*args, **kwargs):
            bot, update = args
            bot.send_chat_action(chat_id=update.effective_message.chat_id, action=action)
            return func(bot, update, **kwargs)

        return command_func

    return decorator


with open('token.txt', 'r') as f:
    token = f.readline()

updater = Updater(token=token)
dispatcher = updater.dispatcher


@send_action(ChatAction.TYPING)
def start_command(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Привет, я помогу тебе распознать номер автомобиля')


@send_action(ChatAction.UPLOAD_PHOTO)
def cat_command(bot, update):
    bot.send_photo(chat_id=update.message.chat_id, photo=open('../images/cat.jpg', 'rb'))


@send_action(ChatAction.TYPING)
def text_message(bot, update):
    response = 'Получил Ваше сообщение: ' + update.message.text
    bot.send_message(chat_id=update.message.chat_id, text=response)


@send_action(ChatAction.UPLOAD_PHOTO)
def image_message(bot, update):
    photo = update.message.photo[-1].file_id
    photo = bot.get_file(photo)
    photo.download('{}.jpg'.format(update.message.photo[-1].file_id))
    result_photo, plate_number = Recognitor.recognize('{}.jpg'.format(update.message.photo[-1].file_id))

    if plate_number and len(plate_number) != 0:
        cv2.imwrite("result.png", result_photo)
        bot.send_photo(chat_id=update.message.chat_id, photo=open('result.png', 'rb'))
        bot.send_message(chat_id=update.message.chat_id, text='Ваш номер автомобиля: {}'.format(plate_number))
    else:
        bot.send_message(chat_id=update.message.chat_id, text='К сожалению не могу найти никакого номера на фотографии')


def main():
    cat_command_handler = CommandHandler('cat', cat_command)
    start_command_handler = CommandHandler('start', start_command)
    text_message_handler = MessageHandler(Filters.text, text_message)
    photo_message_handler = MessageHandler(Filters.photo, image_message)

    dispatcher.add_handler(cat_command_handler)
    dispatcher.add_handler(start_command_handler)
    dispatcher.add_handler(text_message_handler)
    dispatcher.add_handler(photo_message_handler)

    updater.start_polling(clean=True)

    updater.idle()


if __name__ == '__main__':
    main()
