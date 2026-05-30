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

# Título de la aplicación
st.title("🧮 Herramienta de Optimización Completa")
st.markdown("Analiza cualquier tipo de función, muestra el procedimiento algebraico y genera la gráfica profesional detallada.")

# --- ENTRADA DE DATOS CON TRADUCTOR ---
st.info("💡 **Tips de escritura:**\n* Para funciones trigonométricas: `sin(x)`, `cos(x)`.\n* Para raíces/exponentes racionales: Usa paréntesis `x^(1/2)` o usa `sqrt(x)`.\n* Ejemplo por defecto: `2x^3 - 3x2 - 12x + 1`")

default_function = "2*x**3 - 3*x**2 - 12*x + 1"
input_usuario = st.text_input("Ingresa la función f(x):", default_function)

if st.button("Ejecutar Análisis y Graficar"):
    try:
        # --- LIMPIEZA INTELIGENTE ---
        t = input_usuario.lower()
        t = t.replace('sen', 'sin').replace('^', '**')
        # Poner asteriscos faltantes: 3x -> 3*x o 3sin -> 3*sin o 3( -> 3*(
        t = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', t)
        # Arreglar x3 -> x**3 SOLO si no hay un asterisco antes
        t = re.sub(r'([a-zA-Z])(\d)', r'\1**\2', t)
        
        # EL TRUCO AQUÍ: rational=True fuerza a que las fracciones (como 1/2) sean exactas
        f = sp.sympify(t, rational=True)
        
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
                st.write(f"**Evaluando punto en x = {p:.4f}:**")
                
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

            # --- GRÁFICA PROFESIONAL ---
            st.divider()
            st.subheader("📊 Gráfica Profesional Detallada")
            
            x_pts = [r["Punto Crítico (x)"] for r in res_list]
            y_pts = [r["Valor en Y"] for r in res_list]
            margin_x = (max(x_pts) - min(x_pts)) * 0.3 if len(x_pts) > 1 else 3
            x_min, x_max = min(x_pts) - margin_x, max(x_pts) + margin_x
            
            x_arr = np.linspace(float(x_min), float(x_max), 500)
            
            # Lambdify universal
            f_num = sp.lambdify(x, f, modules=['numpy', {'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'log': np.log}])
            y_arr = f_num(x_arr)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            ax.set_title(r"Optimización de $f(x)$", color='maroon', fontsize=14, fontweight='bold')
            
            expr_latex = sp.latex(f)
            ax.plot(x_arr, y_arr, color='blue', label=fr'$f(x) = {expr_latex}$', linewidth=2)
            
            for r in res_list:
                ax.plot(r["Punto Crítico (x)"], r["Valor en Y"], 'ro', markersize=8)
                label_text = f"{r['Clasificación']} ({r['Punto Crítico (x)']:.1f}, {r['Valor en Y']:.1f})"
                
                if r['Clasificación'] == "Máximo":
                    xy_text = (0, 10) 
                else:
                    xy_text = (0, -20)
                
                ax.annotate(label_text, 
                            (r["Punto Crítico (x)"], r["Valor en Y"]), 
                            xytext=xy_text, textcoords='offset points', 
                            fontweight='bold', fontsize=9, color='darkred', ha='center')
            
            ax.axhline(0, color='black', lw=1)
            ax.axvline(0, color='black', lw=1)
            ax.grid(True, alpha=0.3, linestyle='--', color='gray')
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            
            ax.legend(fontsize=10, loc='best')
            
            # Solo ajustar límites en Y si no hay valores infinitos/nan (útil para fracciones)
            valid_y = y_arr[np.isfinite(y_arr)]
            if len(valid_y) > 0:
                y_min_val, y_max_val = min(valid_y), max(valid_y)
                margin_y = (y_max_val - y_min_val) * 0.1 if y_max_val != y_min_val else 1
                ax.set_ylim(y_min_val - margin_y, y_max_val + margin_y)
            
            st.pyplot(fig)

            st.divider()
            st.subheader("📊 Resumen de Resultados Finales")
            df = pd.DataFrame(res_list)
            st.table(df)

    except Exception as e:
        st.error(f"Error al procesar la función. Revisa que esté bien escrita. Error: {e}")