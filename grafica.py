import streamlit as st
import sympy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Configuración de la interfaz
st.set_page_config(page_title="Herramienta de Optimización Completa", layout="centered")

# Definir la variable simbólica
x = sp.Symbol('x')

# Título de la aplicación (esto es el título de Streamlit, no el de la gráfica)
st.title("🧮 Herramienta de Optimización Completa")
st.markdown("Analiza cualquier tipo de función, muestra el procedimiento algebraico y genera la gráfica profesional detallada.")

# --- ENTRADA DE DATOS CON TRADUCTOR ---
st.info("💡 **Tips de escritura:**\n* Para funciones: `sin(x)`, `cos(x)`, `exp(x)`, `log(x)`.\n* Ejemplo por defecto: `2x^3 - 3x2 - 12x + 1`")
# He puesto la función de tu imagen como valor por defecto
default_function = "2*x**3 - 3*x**2 - 12*x + 1"
input_usuario = st.text_input("Ingresa la función f(x):", default_function)

if st.button("Ejecutar Análisis y Graficar"):
    try:
        # --- LIMPIEZA INTELIGENTE ---
        t = input_usuario.lower()
        t = t.replace('sen', 'sin').replace('^', '**')
        # Poner asteriscos faltantes: 3x -> 3*x o 3sin -> 3*sin
        t = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', t)
        # Arreglar x3 -> x**3
        t = re.sub(r'([a-zA-Z])(\d)', r'\1**\2', t)
        
        f = sp.sympify(t)
        
        st.divider()
        
        # --- PASO 1: DERIVADAS ---
        st.subheader("1. Análisis de Derivadas")
        f_p = sp.diff(f, x)
        f_pp = sp.diff(f_p, x)
        
        st.latex(f"f(x) = {sp.latex(f)}")
        st.latex(f"f'(x) = {sp.latex(f_p)}")
        st.latex(f"f''(x) = {sp.latex(f_pp)}")

        # --- PASO 2: PUNTOS CRÍTICOS ---
        puntos = sp.solve(f_p, x)
        puntos_reales = [p.evalf() for p in puntos if p.is_real or p.is_Float]

        if not puntos_reales:
            st.warning("No se encontraron puntos críticos reales en esta función.")
        else:
            # --- PASO 3: PROCEDIMIENTO ---
            st.subheader("2. Sustitución y Clasificación")
            res_list = []
            
            for p in puntos_reales:
                y_val = f.subs(x, p).evalf()
                f2_val = f_pp.subs(x, p).evalf()
                
                if f2_val > 0: tipo = "Mínimo"
                elif f2_val < 0: tipo = "Máximo"
                else: tipo = "Inconcluso"
                
                # Mostrar procedimiento estilo pizarra
                st.write(f"---")
                st.write(f"**Evaluando punto en $x = {p:.4f}$:**")
                
                # Reemplazo visual
                st.latex(f"f({p:.2f}) = {sp.latex(y_val)}")
                st.latex(f"f''({p:.2f}) = {sp.latex(f2_val)}")
                
                # Guardar con los nombres exactos para la tabla
                res_list.append({
                    "Punto Crítico (x)": float(p),
                    "Valor en Y": float(y_val),
                    "f''(x)": float(f2_val),
                    "Clasificación": tipo
                })

            # --- GRÁFICA PROFESIONAL REPLICANDO image_14.png ---
            st.divider()
            st.subheader("📊 Gráfica Profesional Detallada")
            
            # Obtener puntos reales y límites dinámicos para la gráfica
            x_pts = [r["Punto Crítico (x)"] for r in res_list]
            y_pts = [r["Valor en Y"] for r in res_list]
            margin_x = (max(x_pts) - min(x_pts)) * 0.3 if len(x_pts) > 1 else 3
            x_min, x_max = min(x_pts) - margin_x, max(x_pts) + margin_x
            
            x_arr = np.linspace(float(x_min), float(x_max), 500)
            
            # Lambdify universal para funciones trigonométricas, exponenciales, etc.
            f_num = sp.lambdify(x, f, modules=['numpy', {'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'log': np.log}])
            y_arr = f_num(x_arr)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # 1. TÍTULO EN MARRÓN
            ax.set_title(r"Optimización de $f(x)$", color='maroon', fontsize=14, fontweight='bold')
            
            # 2. LÍNEA AZUL SÓLIDA CON LEYENDA LATEX
            expr_latex = sp.latex(f)
            ax.plot(x_arr, y_arr, color='blue', label=fr'$f(x) = {expr_latex}$', linewidth=2)
            
            # 3. PUNTOS ROJOS Y ETIQUETAS ESPECÍFICAS
            for r in res_list:
                # Dibujar punto rojo
                ax.plot(r["Punto Crítico (x)"], r["Valor en Y"], 'ro', markersize=8)
                
                # Crear etiqueta como "Máx (-1, 8)" o "Mín (2, -19)"
                label_text = f"{r['Clasificación']} ({r['Punto Crítico (x)']:.1f}, {r['Valor en Y']:.1f})"
                
                # Ajuste de colocación: Máximo arriba, Mínimo abajo
                if r['Clasificación'] == "Máximo":
                    xy_text = (0, 10) # 10 puntos arriba
                else:
                    xy_text = (0, -20) # 20 puntos abajo
                
                ax.annotate(label_text, 
                            (r["Punto Crítico (x)"], r["Valor en Y"]), 
                            xytext=xy_text, textcoords='offset points', 
                            fontweight='bold', fontsize=9, color='darkred', ha='center')
            
            # 4. EJES Y CUADRÍCULA DASHED
            ax.axhline(0, color='black', lw=1)
            ax.axvline(0, color='black', lw=1)
            ax.grid(True, alpha=0.3, linestyle='--', color='gray')
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            
            # 5. LEYENDA PROFESIONAL
            ax.legend(fontsize=10, loc='best')
            
            # Ajuste de límites dinámicos
            y_min, y_max = min(y_arr), max(y_arr)
            margin_y = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - margin_y, y_max + margin_y)
            
            st.pyplot(fig)

            st.divider()
            st.subheader("📊 Resumen de Resultados Finales")
            df = pd.DataFrame(res_list)
            st.table(df)

    except Exception as e:
        st.error(f"Error al procesar la función. Revisa que esté bien escrita. Error: {e}")