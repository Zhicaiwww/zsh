#!/bin/bash

# 指定要处理的文件路径
file_path="/home/zhicai/.tmux.conf2"

# 定义无效选项列表
invalid_options=(
  "mode-attr"
  "mode-fg"
  "mode-bg"
  "pane-border-bg"
  "pane-border-fg"
  "pane-active-border-bg"
  "pane-active-border-fg"
  "status-attr"
  "window-status-current-fg"
  "window-status-current-bg"
  "window-status-current-attr"
  "window-status-fg"
  "window-status-bg"
  "window-status-attr"
  "window-status-bell-attr"
  "window-status-bell-fg"
  "window-status-bell-bg"
  "message-attr"
  "message-fg"
  "message-bg"
)

# 临时文件路径
temp_file_path="$file_path.tmp"

# 遍历文件的每一行
while IFS= read -r line; do
  # 检查是否包含无效选项
  should_comment=false
  for option in "${invalid_options[@]}"; do
    if [[ $line == *"$option"* ]]; then
      should_comment=true
      break
    fi
  done

  # 注释掉包含无效选项的行
  if [ "$should_comment" = true ]; then
    echo "# $line"
  else
    echo "$line"
  fi
done < "$file_path" > "$temp_file_path"

# 将临时文件替换原始文件
mv "$temp_file_path" "$file_path"

# 删除临时文件（如果存在）
if [ -f "$temp_file_path" ]; then
  rm "$temp_file_path"
fi
 
