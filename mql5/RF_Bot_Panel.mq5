//+------------------------------------------------------------------+
//| RF_Bot_Panel: estado del bot de Python sobre el gráfico de MT5.   |
//|                                                                  |
//| Lee el archivo que publica panel_mt5.py en Common\Files. Es un    |
//| indicador, y MT5 no permite a los indicadores enviar órdenes.     |
//|                                                                  |
//| Instalación: copiarlo a <carpeta de datos>\MQL5\Indicators,       |
//| compilarlo y arrastrarlo desde el Navegador al gráfico.           |
//+------------------------------------------------------------------+
#property description "Estado del bot de Python. Solo lectura: no puede enviar órdenes."
#property version     "1.00"
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

input int InpX = 10;  // Distancia al borde izquierdo (px)
input int InpY = 30;  // Distancia al borde superior (px)

#define ARCHIVO        "rf_bot_panel.txt"
#define PREFIJO        "RFPANEL_"
#define MAX_LATIDO_S   20    // Python publica al menos cada 5 s
#define ANCHO          350
#define ALTO           242
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

int E(const int valor) { return (int)MathRound(valor * g_escala); }

//--- Lectura. Si el archivo no se puede abrir (Python lo está reemplazando) se conserva la lectura anterior.
void LeerEstado()
{
   const int h = FileOpen(ARCHIVO, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ | FILE_SHARE_WRITE, '\t', CP_UTF8);
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
   ObjectSetInteger(0, n, OBJPROP_XDISTANCE, InpX + E(x));
   ObjectSetInteger(0, n, OBJPROP_YDISTANCE, InpY + E(y));
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
   ObjectSetInteger(0, n, OBJPROP_XDISTANCE, InpX + E(x));
   ObjectSetInteger(0, n, OBJPROP_YDISTANCE, InpY + E(y));
   ObjectSetInteger(0, n, OBJPROP_COLOR, c);
   ObjectSetInteger(0, n, OBJPROP_FONTSIZE, tam);
   ObjectSetString(0, n, OBJPROP_FONT, fuente);
   ObjectSetString(0, n, OBJPROP_TEXT, texto == "" ? " " : texto);  // un texto vacío se mostraría como "Label"
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

void Actualizar()
{
   LeerEstado();
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

   Caja("fondo", 0, 0, ANCHO, ALTO, C_FONDO, C_BORDE);
   Texto("titulo", MARGEN, 10, "RF Bot  " + Valor("simbolo"), C_TEXTO, 10, "Segoe UI Semibold");
   Texto("estado", ANCHO - MARGEN, 11, estado, c_estado, 9, "Segoe UI Semibold", ANCHOR_RIGHT_UPPER);
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

//--- Eventos
int OnInit()
{
   g_escala = TerminalInfoInteger(TERMINAL_SCREEN_DPI) / 96.0;
   IndicatorSetString(INDICATOR_SHORTNAME, "RF Bot Panel");
   EventSetTimer(1);
   Actualizar();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   ObjectsDeleteAll(0, PREFIJO);
   ChartRedraw();
}

void OnTimer() { Actualizar(); }

int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
{
   return rates_total;
}
