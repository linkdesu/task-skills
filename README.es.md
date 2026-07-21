# task-skills

Este repositorio es un catalogo multi-skill para `npx skills add`.

Muchas skills de este repositorio nacen de tareas complejas y puntuales, con alta dependencia del entorno. Antes, este tipo de soluciones se compartia como scripts de shell o Python independientes. Esos scripts ayudaron a muchas personas, pero tambien consumian mucho tiempo de desarrolladores voluntarios. Este repositorio mantiene ese espiritu, pero usando skills de AI como formato de intercambio.

Hasta que los dispositivos personales puedan alcanzar facilmente la capacidad de los modelos SOTA actuales, compartir skills practicas sigue siendo el camino mas eficiente. El objetivo a largo plazo es claro: convertir experiencia practica en skills reutilizables, para que mas personas puedan resolver problemas dificiles de build y configuracion mas rapido con modelos eficientes.

## Translation Links

- English: [README.md](README.md)
- Chinese: [README.zh-CN.md](README.zh-CN.md)
- Japanese: [README.ja.md](README.ja.md)

## Comandos de instalacion

No suele ser buena idea instalar todas las skills de una vez, porque desperdicia contexto. El flujo recomendado es que tu AI lea este README, seleccione solo la skill relevante para la tarea actual y la instale a nivel de proyecto.

Instalar una skill por nombre:

```bash
npx skills add <owner>/<repo> --skill <skill>
```

Instalar una skill por ruta directa:

```bash
npx skills add https://github.com/<owner>/<repo>/tree/main/skills/<skill>
```

## Indice de Skills

- build-sageattention-rocm-on-win11: [skills/build-sageattention-rocm-on-win11/SKILL.md](skills/build-sageattention-rocm-on-win11/SKILL.md)

## Como contribuir una nueva skill

1. Usa un modelo de AI capaz para resolver tu tarea real de principio a fin.
2. Pidele que resuma el flujo exitoso como una skill reutilizable.
3. Guarda esa skill en `skills/<your-skill-name>/`.
4. Actualiza el indice para que otros agentes puedan encontrarla rapido.

## Notas

- Prioriza instalar por tarea en lugar de instalar en bloque.
- Manten cada skill enfocada en un problema concreto.
- Incluye, cuando sea posible, errores conocidos y pasos de verificacion.
