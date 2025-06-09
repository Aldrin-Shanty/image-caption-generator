#!/bin/sh
chosen=$(echo -e "Lock\nShutdown\nReboot\nLogout" | rofi -dmenu -i -p "Power Menu" -lines 4 \
    -theme-str 'window {width: 400px;} listview {lines: 4; spacing: 0;} element {font: "monospace 30"; padding: 15px;} entry {enabled: false;}')

case "$chosen" in
    Lock)      dm-tool switch-to-greeter ;;
    Shutdown)  systemctl poweroff ;;
    Reboot)    systemctl reboot ;;
    Logout)    bspc quit ;;
esac
