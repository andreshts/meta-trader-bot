//+------------------------------------------------------------------+
//| RF_Bot_Panel: estado del bot de Python sobre el gráfico de MT5.   |
//|                                                                  |
//| Lee el archivo que publica panel_mt5.py en Common\Files. Es un    |
//| indicador, y MT5 no permite a los indicadores enviar órdenes.     |
//|                                                                  |
//| Instalación: copiarlo a <carpeta de datos>\MQL5\Indicators,       |
//| compilarlo y arrastrarlo desde el Navegador al gráfico.           |
//|                                                                  |
//| Uso: se mueve arrastrándolo desde cualquier parte; los botones    |
//| lo minimizan o lo quitan del gráfico (el bot sigue funcionando).  |
//+------------------------------------------------------------------+
#property description "Estado del bot de Python. Solo lectura: no puede enviar órdenes."
#property version     "1.11"
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

#define NOMBRE_CORTO   "RF Bot Panel"
#define ARCHIVO        "rf_bot_panel.txt"
#define PREFIJO        "RFPANEL_"
#define MAX_LATIDO_S   20    // Python publica al menos cada 5 s
#define X_INICIAL      10
#define Y_INICIAL      30
#define ANCHO          350
#define ALTO           242
#define ALTO_MINIMIZADO 34
#define LADO_BOTON     20
#define X_CERRAR       (ANCHO - 28)
#define X_MINIMIZAR    (ANCHO - 52)
#define Y_BOTON        7
#define MARGEN         12
#define X_BARRA        112
#define ANCHO_BARRA    110
#define MAX_CARACTERES 52
#define LINEAS_NOTA    3

#define C_FONDO C'22,27,34'
#define C_BORDE C'48,54,61'
#define C_TEXTO C'230,237,243'
#define C_TENUE C'139,148,158'
#define C_VERDE C'63,185,80'
#define C_AMBAR C'210,153,34'
#define C_ROJO  C'248,81,73'
#define C_AZUL  C'88,166,255'

string g_claves[];
string g_valores[];
double g_escala = 1.0;
int g_x = X_INICIAL, g_y = Y_INICIAL;
bool g_minimizado = false;
bool g_pulsado = false, g_arrastrando = false, g_sobre_panel = false;
int g_agarre_x = 0, g_agarre_y = 0;
bool g_scroll_original = true, g_eventos_original = false;  // ajustes del gráfico que se restauran al salir

int E(const int valor) { return (int)MathRound(valor * g_escala); }

int AltoActual() { return g_minimizado ? ALTO_MINIMIZADO : ALTO; }

bool Dentro(const int x, const int y, const int izquierda, const int arriba, const int ancho, const int alto)
{
   return x >= g_x + E(izquierda) && x < g_x + E(izquierda + ancho) && y >= g_y + E(arriba) && y < g_y + E(arriba + alto);
}

// Los botones siempre quedan a la vista, aunque se encoja la ventana o se arrastre el panel fuera.
void LimitarPosicion()
{
   g_x = MathMax(0, MathMin(g_x, (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS) - E(ANCHO)));
   g_y = MathMax(0, MathMin(g_y, (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS, 0) - E(ALTO_MINIMIZADO)));
}

//--- Posición por gráfico: un indicador se reinicia al cambiar de símbolo o de temporalidad y la perdería.
string ClavePosicion(const string campo = "") { return PREFIJO + IntegerToString(ChartID()) + "_" + campo; }

void GuardarPosicion()
{
   GlobalVariableSet(ClavePosicion("x"), g_x);
   GlobalVariableSet(ClavePosicion("y"), g_y);
   GlobalVariableSet(ClavePosicion("minimizado"), g_minimizado);
}

void CargarPosicion()
{
   if(!GlobalVariableCheck(ClavePosicion("x")))
      return;
   g_x = (int)GlobalVariableGet(ClavePosicion("x"));
   g_y = (int)GlobalVariableGet(ClavePosicion("y"));
   g_minimizado = GlobalVariableGet(ClavePosicion("minimizado")) != 0;
}

//--- Cada gráfico muestra el bot de su símbolo; si ese símbolo no tiene bot propio, el del archivo general.
string ArchivoPanel()
{
   const string propio = "rf_bot_panel_" + _Symbol + ".txt";
   return FileIsExist(propio, FILE_COMMON) ? propio : ARCHIVO;
}

//--- Lectura. Si el archivo no se puede abrir (Python lo está reemplazando) se conserva la lectura anterior.
void LeerEstado()
{
   const int h = FileOpen(ArchivoPanel(), FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ | FILE_SHARE_WRITE, '\t', CP_UTF8);
   if(h == INVALID_HANDLE)
      return;
   int n = 0;
   ArrayResize(g_claves, 0);
   ArrayResize(g_valores, 0);
   while(!FileIsEnding(h))
   {
      const string linea = FileReadString(h);
      const int igual = StringFind(linea, "=");
      if(igual <= 0)
         continue;
      ArrayResize(g_claves, n + 1);
      ArrayResize(g_valores, n + 1);
      g_claves[n] = StringSubstr(linea, 0, igual);
      g_valores[n] = StringSubstr(linea, igual + 1);
      n++;
   }
   FileClose(h);
}

string Valor(const string clave)
{
   for(int i = 0; i < ArraySize(g_claves); i++)
      if(g_claves[i] == clave)
         return g_valores[i];
   return "";
}

//--- Dibujo
void Caja(const string nombre, const int x, const int y, const int ancho, const int alto, const color fondo, const color borde)
{
   const string n = PREFIJO + nombre;
   if(ObjectFind(0, n) < 0)
   {
      ObjectCreate(0, n, OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ObjectSetInteger(0, n, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, n, OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, n, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, n, OBJPROP_HIDDEN, true);
   }
   ObjectSetInteger(0, n, OBJPROP_XDISTANCE, g_x + E(x));
   ObjectSetInteger(0, n, OBJPROP_YDISTANCE, g_y + E(y));
   ObjectSetInteger(0, n, OBJPROP_XSIZE, E(ancho));
   ObjectSetInteger(0, n, OBJPROP_YSIZE, E(alto));
   ObjectSetInteger(0, n, OBJPROP_BGCOLOR, fondo);
   ObjectSetInteger(0, n, OBJPROP_COLOR, borde);
}

void Texto(const string nombre, const int x, const int y, const string texto, const color c, const int tam = 9,
           const string fuente = "Segoe UI", const ENUM_ANCHOR_POINT ancla = ANCHOR_LEFT_UPPER)
{
   const string n = PREFIJO + nombre;
   if(ObjectFind(0, n) < 0)
   {
      ObjectCreate(0, n, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, n, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, n, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, n, OBJPROP_HIDDEN, true);
   }
   ObjectSetInteger(0, n, OBJPROP_ANCHOR, ancla);
   ObjectSetInteger(0, n, OBJPROP_XDISTANCE, g_x + E(x));
   ObjectSetInteger(0, n, OBJPROP_YDISTANCE, g_y + E(y));
   ObjectSetInteger(0, n, OBJPROP_COLOR, c);
   ObjectSetInteger(0, n, OBJPROP_FONTSIZE, tam);
   ObjectSetString(0, n, OBJPROP_FONT, fuente);
   ObjectSetString(0, n, OBJPROP_TEXT, texto == "" ? " " : texto);  // un texto vacío se mostraría como "Label"
}

void Boton(const string nombre, const int x, const string simbolo)
{
   Caja("boton_" + nombre, x, Y_BOTON, LADO_BOTON, LADO_BOTON, C_BORDE, C_BORDE);
   Texto("boton_" + nombre + "_simbolo", x + LADO_BOTON / 2, Y_BOTON + LADO_BOTON / 2, simbolo, C_TEXTO, 10, "Segoe UI Semibold", ANCHOR_CENTER);
}

// La barra llena significa que la probabilidad ha llegado a su umbral.
void Barra(const string nombre, const int y, const string etiqueta, const string p, const string umbral)
{
   const double prob = StringToDouble(p), limite = StringToDouble(umbral);
   const bool calculable = p != "" && umbral != "" && limite > 0;
   const bool llega = calculable && prob >= limite;
   const double fraccion = calculable ? MathMin(prob / limite, 1.0) : 0.0;

   string valor = (p == "" ? "--" : p) + " / " + umbral;
   if(umbral == "")
      valor = ArraySize(g_claves) > 0 ? "desactivado" : "--";

   Texto(nombre + "_etiqueta", MARGEN, y, etiqueta, C_TENUE);
   Caja(nombre + "_pista", X_BARRA, y + 5, ANCHO_BARRA, 8, C_BORDE, C_BORDE);
   const color relleno = llega ? C_VERDE : C_AZUL;
   Caja(nombre + "_relleno", X_BARRA, y + 5, MathMax((int)MathRound(ANCHO_BARRA * fraccion), 1), 8, relleno, relleno);
   Texto(nombre + "_valor", ANCHO - MARGEN, y, valor, llega ? C_VERDE : C_TEXTO, 9, "Segoe UI", ANCHOR_RIGHT_UPPER);
}

string TextoPosicion(const long magic, const string simbolo, color &c)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetTicket(i) == 0 || PositionGetInteger(POSITION_MAGIC) != magic || PositionGetString(POSITION_SYMBOL) != simbolo)
         continue;
      const double resultado = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
      c = resultado >= 0 ? C_VERDE : C_ROJO;
      return StringFormat("%s %.2f a %s  |  %+.2f %s",
                          PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? "Compra" : "Venta",
                          PositionGetDouble(POSITION_VOLUME),
                          DoubleToString(PositionGetDouble(POSITION_PRICE_OPEN), (int)SymbolInfoInteger(simbolo, SYMBOL_DIGITS)),
                          resultado, AccountInfoString(ACCOUNT_CURRENCY));
   }
   c = C_TEXTO;
   return "ninguna";
}

void PartirNota(const string nota, string &lineas[])
{
   ArrayResize(lineas, LINEAS_NOTA);
   for(int i = 0; i < LINEAS_NOTA; i++)
      lineas[i] = "";
   string palabras[];
   const int n = StringSplit(nota, ' ', palabras);
   int l = 0;
   for(int i = 0; i < n; i++)
   {
      const string palabra = StringSubstr(palabras[i], 0, MAX_CARACTERES);
      if(lineas[l] != "" && StringLen(lineas[l]) + StringLen(palabra) + 1 > MAX_CARACTERES)
      {
         if(l == LINEAS_NOTA - 1)
         {
            lineas[l] += " ...";
            return;
         }
         l++;
      }
      lineas[l] += (lineas[l] == "" ? "" : " ") + palabra;
   }
}

void Dibujar()
{
   LimitarPosicion();
   const bool hay_datos = ArraySize(g_claves) > 0;

   string estado = "SIN DATOS";
   color c_estado = C_TENUE;
   if(hay_datos)
   {
      const long edad = (long)TimeGMT() - StringToInteger(Valor("actualizado"));
      if(Valor("estado") == "detenido")
         estado = "DETENIDO";
      else if(edad > MAX_LATIDO_S)
      {
         estado = "SIN SEÑAL DEL BOT";
         c_estado = C_ROJO;
      }
      else
      {
         estado = "EN MARCHA";
         c_estado = C_VERDE;
      }
   }

   Caja("fondo", 0, 0, ANCHO, AltoActual(), C_FONDO, C_BORDE);
   Texto("titulo", MARGEN, 10, "RF Bot  " + Valor("simbolo"), C_TEXTO, 10, "Segoe UI Semibold");
   Texto("estado", X_MINIMIZAR - 8, 11, estado, c_estado, 9, "Segoe UI Semibold", ANCHOR_RIGHT_UPPER);
   Boton("minimizar", X_MINIMIZAR, g_minimizado ? "+" : "–");
   Boton("cerrar", X_CERRAR, "×");
   if(g_minimizado)
   {
      ChartRedraw();
      return;
   }

   Texto("cuenta", MARGEN, 31, hay_datos ? "Cuenta " + Valor("cuenta") + "  |  Termina a las " + Valor("fin") : "", C_TENUE);
   Caja("linea_1", MARGEN, 53, ANCHO - 2 * MARGEN, 1, C_BORDE, C_BORDE);

   Barra("long", 62, "Compra (long)", Valor("p_long"), Valor("umbral_long"));
   Barra("short", 84, "Venta (short)", Valor("p_short"), Valor("umbral_short"));
   const string evaluada = Valor("evaluada");
   Texto("evaluada", MARGEN, 106, evaluada == "" ? "Ninguna vela evaluada todavía" : "Última vela evaluada a las " + evaluada, C_TENUE);
   Caja("linea_2", MARGEN, 129, ANCHO - 2 * MARGEN, 1, C_BORDE, C_BORDE);

   color c_posicion;
   const string posicion = TextoPosicion(StringToInteger(Valor("magic")), Valor("simbolo"), c_posicion);
   Texto("posicion_etiqueta", MARGEN, 138, "Posición", C_TENUE);
   Texto("posicion_valor", ANCHO - MARGEN, 138, posicion, c_posicion, 9, "Segoe UI", ANCHOR_RIGHT_UPPER);
   Caja("linea_3", MARGEN, 161, ANCHO - 2 * MARGEN, 1, C_BORDE, C_BORDE);

   const string nivel = Valor("nota_nivel");
   const color c_nota = (nivel == "ERROR" || nivel == "CRITICAL") ? C_ROJO : (nivel == "WARNING" ? C_AMBAR : C_TEXTO);
   const string hora = Valor("nota_hora");
   Texto("nota_etiqueta", MARGEN, 170, "Última decisión" + (hora == "" ? "" : " (" + hora + ")"), C_TENUE);
   string lineas[];
   PartirNota(hay_datos ? Valor("nota") : "Se activa al lanzar el bot con demo_exness_bot.py", lineas);
   for(int i = 0; i < LINEAS_NOTA; i++)
      Texto("nota_" + IntegerToString(i), MARGEN, 189 + i * 16, lineas[i], c_nota);

   ChartRedraw();
}

void Actualizar()
{
   LeerEstado();
   Dibujar();
}

void AlternarMinimizado()
{
   g_minimizado = !g_minimizado;
   GuardarPosicion();
   ObjectsDeleteAll(0, PREFIJO);  // se recrea en orden para que el fondo quede debajo del resto
   Dibujar();
}

//--- Eventos
int OnInit()
{
   g_escala = TerminalInfoInteger(TERMINAL_SCREEN_DPI) / 96.0;
   IndicatorSetString(INDICATOR_SHORTNAME, NOMBRE_CORTO);
   g_scroll_original = (bool)ChartGetInteger(0, CHART_MOUSE_SCROLL);
   g_eventos_original = (bool)ChartGetInteger(0, CHART_EVENT_MOUSE_MOVE);
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, true);
   CargarPosicion();
   EventSetTimer(1);
   Actualizar();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   ChartSetInteger(0, CHART_MOUSE_SCROLL, g_scroll_original);
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, g_eventos_original);
   if(reason == REASON_REMOVE || reason == REASON_CHARTCLOSE)
      GlobalVariablesDeleteAll(ClavePosicion());
   ObjectsDeleteAll(0, PREFIJO);
   ChartRedraw();
}

void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_CHART_CHANGE)
   {
      Dibujar();
      return;
   }
   if(id != CHARTEVENT_MOUSE_MOVE)
      return;

   const int x = (int)lparam, y = (int)dparam;
   const bool pulsado = ((uint)StringToInteger(sparam) & 1) != 0;  // bit 1: botón izquierdo
   const bool pulsacion = pulsado && !g_pulsado;
   g_pulsado = pulsado;

   if(g_arrastrando)
   {
      if(pulsado)
      {
         g_x = x - g_agarre_x;
         g_y = y - g_agarre_y;
         Dibujar();
         return;
      }
      g_arrastrando = false;
      GuardarPosicion();
   }

   const bool sobre_panel = Dentro(x, y, 0, 0, ANCHO, AltoActual());
   if(!pulsado && sobre_panel != g_sobre_panel)  // sobre el panel, arrastrar no debe desplazar el gráfico
   {
      g_sobre_panel = sobre_panel;
      ChartSetInteger(0, CHART_MOUSE_SCROLL, !sobre_panel && g_scroll_original);
   }
   if(!pulsacion || !sobre_panel)
      return;

   if(Dentro(x, y, X_CERRAR, Y_BOTON, LADO_BOTON, LADO_BOTON))
      ChartIndicatorDelete(0, 0, NOMBRE_CORTO);
   else if(Dentro(x, y, X_MINIMIZAR, Y_BOTON, LADO_BOTON, LADO_BOTON))
      AlternarMinimizado();
   else
   {
      g_arrastrando = true;
      g_agarre_x = x - g_x;
      g_agarre_y = y - g_y;
   }
}

void OnTimer() { Actualizar(); }

int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
{
   return rates_total;
}
