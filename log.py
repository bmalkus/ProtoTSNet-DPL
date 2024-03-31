def create_logger(log_filename, display=True):
    f = open(log_filename, 'a')
    logger_display = display

    def logger(text, flush=False, display=logger_display):
        if display:
            print(text)
        f.write(text + '\n')
        if flush:
            f.flush()

    return logger, f.close
