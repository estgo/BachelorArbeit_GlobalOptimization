import ttkbootstrap as ttkb



def create_button(parent, text, x=0, y=0, command=None, state='normal'):
    button = ttkb.Button(parent, text=text, command=command)
    button.place(x=x, y=y)
    button.config(state=state)
    return button


def create_combobox(parent, options, x=0, y=0, width=20, default_text="Select an Option", command=None,
                    state='readonly'):
    combobox = ttkb.Combobox(parent, values=options, state=state, width=width)
    combobox.set(default_text)
    combobox.place(x=x, y=y)

    if command:
        combobox.bind("<<ComboboxSelected>>", command)
    return combobox


