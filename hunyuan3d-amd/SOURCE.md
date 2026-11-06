# Источник и состав

Пакет основан на исходном AMD-установщике владельца репозитория:

- `installer/Install-AMD.ps1` — функции `InstallAmdComfyDeps`, `InstallAmdComfy`, загрузка Hunyuan3D-2 и u2net;
- `installer/amd/AmdCore.ps1` — без изменений, исходный blob SHA `793b22bee8e7dc4e73eefc4303b09d4f35917c5b`.

`installer/Install-Hunyuan3D-AMD.ps1` — самодостаточная точка входа. Исходные AMD-функции сохранены; Pixal3D/WSL удалены, небольшие общие функции загрузки и распаковки включены напрямую, добавлен `extra_model_paths.yaml`. Внешние общие модули не требуются.
