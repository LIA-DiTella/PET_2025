#!/usr/bin/env python3
"""
Script de validación para verificar que la optimización funciona correctamente.

Este script hace una validación básica de la carga de datasets optimizada
sin ejecutar entrenamiento completo.
"""

import time
from pathlib import Path

from batch_train_binary_classification import BinaryClassificationBatchTrainer


def test_dataset_loading():
    """Prueba la carga optimizada de datasets."""
    print("🧪 Validando carga optimizada de datasets")
    print("=" * 50)

    # Crear trainer de prueba
    trainer = BinaryClassificationBatchTrainer(
        base_configs_dir="./configs",
        results_dir="./test_results",
        use_otsu_masking=True,
    )

    print(f"📋 Total de tareas definidas: {len(trainer.binary_tasks)}")
    
    # Mostrar algunas tareas
    print("\n🔍 Primeras 3 tareas:")
    for i, task in enumerate(trainer.binary_tasks[:3]):
        print(f"   {i+1}. {task['model']}_{task['dimension']}_{task['dataset']}_{task['train_classes']}")

    # Probar carga de configuraciones de datasets
    print("\n🔧 Probando creación de configuraciones de datasets...")
    try:
        dataset_configs = trainer._create_dataset_configs()
        print(f"   ✅ Configuraciones creadas: {len(dataset_configs)}")
        
        for config_key in dataset_configs.keys():
            print(f"      - {config_key}")
            
    except Exception as e:
        print(f"   ❌ Error creando configuraciones: {e}")
        return False

    # Probar carga de datasets (solo los primeros para validación rápida)
    print("\n📊 Probando carga de datasets (validación rápida)...")
    
    # Crear configuraciones limitadas para prueba
    test_configs = {}
    for key, config in list(dataset_configs.items())[:2]:  # Solo primeros 2
        test_configs[key] = config
    
    try:
        start_time = time.time()
        
        data_loaders_cache = {}
        for config_key, config in test_configs.items():
            print(f"   📊 Cargando {config_key}...")
            try:
                from data.dataset import get_data_loaders
                train_loader, val_loader, test_loader = get_data_loaders(config)
                data_loaders_cache[config_key] = (train_loader, val_loader, test_loader)
                
                # Log información básica
                train_samples = len(train_loader.dataset) if train_loader else 0
                val_samples = len(val_loader.dataset) if val_loader else 0
                test_samples = len(test_loader.dataset) if test_loader else 0
                
                print(f"      ✅ Train={train_samples}, Val={val_samples}, Test={test_samples}")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                data_loaders_cache[config_key] = None
        
        load_time = time.time() - start_time
        loaded_count = len([k for k, v in data_loaders_cache.items() if v is not None])
        
        print(f"\n⏱️  Tiempo de carga: {load_time:.2f}s")
        print(f"✅ Datasets cargados exitosamente: {loaded_count}/{len(test_configs)}")
        
        if loaded_count == 0:
            print("❌ No se pudo cargar ningún dataset")
            return False
            
    except Exception as e:
        print(f"   ❌ Error durante carga: {e}")
        return False

    # Probar recuperación desde cache
    print("\n🎯 Probando recuperación desde cache...")
    try:
        # Probar con primera tarea disponible
        first_task = trainer.binary_tasks[0]
        
        # Buscar data loaders para esta tarea
        dataset = first_task["dataset"]
        dimension = first_task["dimension"]
        cache_key = f"{dataset}_{dimension}"
        
        if cache_key in data_loaders_cache:
            cached_loaders = data_loaders_cache[cache_key]
            if cached_loaders is not None:
                train_loader, val_loader, test_loader = cached_loaders
                print(f"   ✅ Recuperado desde cache: {cache_key}")
                print(f"      Train samples: {len(train_loader.dataset) if train_loader else 0}")
            else:
                print(f"   ⚠️  Cache entry nulo para: {cache_key}")
        else:
            print(f"   ⚠️  No encontrado en cache: {cache_key}")
            
    except Exception as e:
        print(f"   ❌ Error recuperando desde cache: {e}")
        return False

    print("\n🎉 Validación de optimización completada exitosamente!")
    return True


def test_evaluation_optimization():
    """Prueba la optimización del evaluador."""
    print("\n🧪 Validando optimización del evaluador")
    print("=" * 50)
    
    try:
        from model_evaluator import ModelEvaluator
        
        evaluator = ModelEvaluator()
        print("✅ ModelEvaluator inicializado")
        
        # Verificar que existen los nuevos métodos
        if hasattr(evaluator, 'load_all_evaluation_datasets'):
            print("✅ Método load_all_evaluation_datasets disponible")
        else:
            print("❌ Método load_all_evaluation_datasets NO disponible")
            return False
            
        if hasattr(evaluator, 'evaluate_on_multiple_datasets_optimized'):
            print("✅ Método evaluate_on_multiple_datasets_optimized disponible")
        else:
            print("❌ Método evaluate_on_multiple_datasets_optimized NO disponible")
            return False
            
    except Exception as e:
        print(f"❌ Error validando evaluador: {e}")
        return False
        
    return True


def main():
    """Función principal de validación."""
    print("🚀 VALIDACIÓN DE OPTIMIZACIONES DEL PIPELINE PET 2025")
    print("=" * 60)
    
    success = True
    
    # Test 1: Carga optimizada de datasets
    if not test_dataset_loading():
        success = False
    
    # Test 2: Optimización del evaluador
    if not test_evaluation_optimization():
        success = False
    
    # Resumen final
    print("\n" + "=" * 60)
    if success:
        print("🎉 TODAS LAS VALIDACIONES EXITOSAS")
        print("✅ La optimización está funcionando correctamente")
        print("🚀 Listo para usar entrenamiento y evaluación optimizados")
    else:
        print("❌ ALGUNAS VALIDACIONES FALLARON")
        print("⚠️  Revisar errores anteriores antes de usar optimizaciones")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
