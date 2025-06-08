# JSON Specification for Bodies and Genotypes

This document describes the JSON schema used to define creatures in the
simulation engine. A **phenotype** describes a fully realized body with all
rigid bodies and joints specified. A **genotype** is a higher level blueprint
that can expand into a phenotype.

## Schema Overview

A phenotype JSON file is an object with the following top level fields:

```json
{
  "name": "unique creature name",
  "version": "0.1",
  "bodies": [...],
  "joints": [...],
  "actuators": [...]
}
```

* `name` – human readable identifier
* `version` – schema version
* `bodies` – array of rigid body definitions
* `joints` – array connecting bodies together
* `actuators` – (optional) motors controlling joints

### Bodies

Each entry in `bodies` specifies a rigid body. Minimal fields are:

```json
{
  "id": "torso",
  "shape": "box", // box | sphere | capsule | etc.
  "dimensions": [0.5, 1.0, 0.3],
  "mass": 10.0,
  "material": { "friction": 0.5, "restitution": 0.1 }
}
```

### Joints

Joints connect a parent body to a child body and restrict relative motion:

```json
{
  "id": "hip_left",
  "parent": "torso",
  "child": "leg_left",
  "type": "revolute", // fixed | revolute | prismatic
  "axis": [1, 0, 0],
  "limits": { "lower": -1.57, "upper": 1.57 }
}
```

### Actuators

Actuators drive joints and define the input channels for control:

```json
{
  "joint": "hip_left",
  "strength": 30.0,
  "damping": 0.1
}
```

## Genotype Structure

A genotype describes how to build a phenotype from reusable modules. The format
is intentionally lightweight so different algorithms can modify or evolve it.

```json
{
  "name": "biped_genotype",
  "modules": [
    { "id": "leg", "count": 2, "template": { ... } }
  ],
  "connections": [
    { "from": "torso", "to": "leg", "joint": "hip" }
  ]
}
```

* `modules` – reusable building blocks that may be instanced multiple times
* `connections` – how modules are attached together

A genotypic module expands into one or more bodies and joints when converted to
a phenotype. The exact expansion logic will be implemented later.

## Example Phenotype

```json
{
  "name": "simple_bot",
  "version": "0.1",
  "bodies": [
    { "id": "torso", "shape": "box", "dimensions": [0.5, 1.0, 0.3], "mass": 10.0 },
    { "id": "leg_left", "shape": "capsule", "dimensions": [0.1, 0.5], "mass": 2.0 }
  ],
  "joints": [
    {
      "id": "hip_left",
      "parent": "torso",
      "child": "leg_left",
      "type": "revolute",
      "axis": [1, 0, 0],
      "limits": { "lower": -1.57, "upper": 1.57 }
    }
  ],
  "actuators": [
    { "joint": "hip_left", "strength": 30.0, "damping": 0.1 }
  ]
}
```

## Example Genotype

```json
{
  "name": "biped_genotype",
  "modules": [
    { "id": "leg", "count": 2, "template": { "bodies": [...], "joints": [...] } }
  ],
  "connections": [
    { "from": "torso", "to": "leg", "joint": "hip" }
  ]
}
```

This documentation will grow as the engine evolves, but these initial examples
capture the structure of JSON files used to describe bodies in the system.
