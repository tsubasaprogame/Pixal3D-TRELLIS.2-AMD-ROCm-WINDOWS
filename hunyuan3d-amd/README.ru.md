# Hunyuan3D-2 для AMD ROCm / Windows

Самодостаточный установочный комплект для **ComfyUI + Hunyuan3D-2 + u2net**: Pixal3D и WSL не устанавливаются, внешние общие модули не требуются.

## Что делает установщик

- скачивает ComfyUI v0.33.2 в `runtime\ComfyUI`;
- создаёт отдельное окружение Python 3.12 через `uv`;
- устанавливает ROCm 7.2.1 и PyTorch 2.9.1+rocm7.2.1 с `repo.radeon.com`;
- проверяет, что PyTorch действительно использует HIP и AMD GPU;
- ставит библиотеки ComfyUI, удаления фона и обработки мешей;
- скачивает Hunyuan3D-2 (`hunyuan3d-dit-v2.safetensors`, около 4,9 ГБ) и u2net;
- создаёт `extra_model_paths.yaml`, чтобы ComfyUI видел модель в `data\models\checkpoints`.

**Целевая конфигурация: AMD Radeon RX 9070 XT (`gfx1201`).** Проверка `Assert-AmdTorch` намеренно требует именно RX 9070 XT. Другие GPU не заявляются как поддерживаемые. Автор сообщает об успешном запуске и работе Hunyuan3D на AMD; точные автоматические аппаратные тесты и замеры не проводились.

Весов моделей и скачанного ComfyUI в Git нет — они загружаются при установке.

## Установка

1. Скопируйте каталог `hunyuan3d-amd` в новый короткий путь латиницей, например `C:\AMD-AI\Hunyuan3D`. Не распаковывайте поверх NVIDIA-окружения.
2. Для проверки плана без скачиваний запустите:

   ```bat
   "Install Hunyuan3D AMD.cmd" -DryRun
   ```

3. Для установки запустите `Install Hunyuan3D AMD.cmd` или выполните:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File installer\Install-Hunyuan3D-AMD.ps1
   ```

4. После установки запустите `Start ComfyUI AMD.cmd`. Откроется `http://127.0.0.1:8190`.

Повторный запуск продолжит прерванные загрузки и пропустит уже скачанные файлы.

## Важно

- Установщик не заменяет драйверы и не требует WSL.
- Для генерации в ComfyUI используйте узлы Hunyuan3D и модель `hunyuan3d-dit-v2.safetensors`.
- Модели Hunyuan3D распространяются по **Tencent Hunyuan Community License**. В исходной документации указано ограничение использования на территории ЕС, Великобритании и Южной Кореи; проверьте актуальный текст лицензии перед использованием.
- Сторонние компоненты сохраняют собственные лицензии. Смотрите корневые [`LICENSE`](../LICENSE) и [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Проверка файлов

На Windows с PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\Hunyuan3D-AMD.test.ps1
```

Это статические unit-тесты конфигурации, а не генерация на GPU.
