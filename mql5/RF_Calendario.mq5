//+------------------------------------------------------------------+
//| RF_Calendario: exporta el calendario económico de MT5 a          |
//| Common\Files\rf_calendario.csv para el bot de Python, que no     |
//| tiene acceso al calendario desde su API.                         |
//|                                                                  |
//| Es un servicio: copiar a MQL5\Services, compilar y, en el        |
//| Navegador de la terminal, Servicios -> Añadir servicio -> Iniciar.|
//| Arranca solo con la terminal y vuelve a exportar cada hora.      |
//| Solo lee el calendario y escribe ese archivo: no opera.          |
//+------------------------------------------------------------------+
#property service
#property copyright "RF Bot"
#property version   "1.00"
#property description "Exporta el calendario económico a Common\\Files\\rf_calendario.csv"

input string Monedas                   = "USD,EUR"; // Monedas, separadas por comas
input int    DiasDeHistoria            = 2000;      // Días hacia atrás (para medir su efecto en el histórico)
input int    DiasFuturos               = 14;        // Días hacia delante
input int    ImportanciaMinima         = 2;         // 1 = baja, 2 = media, 3 = alta
input int    MinutosEntreExportaciones = 60;

const string ARCHIVO  = "rf_calendario.csv";
const string TEMPORAL = "rf_calendario.tmp";

//+------------------------------------------------------------------+
void OnStart()
  {
   while(!IsStopped())
     {
      if(Exportar())
         Print("Calendario exportado a Common\\Files\\", ARCHIVO);
      for(int s = 0; s < MinutosEntreExportaciones * 60 && !IsStopped(); s++)
         Sleep(1000);
     }
  }

//+------------------------------------------------------------------+
//| Las horas del calendario vienen en hora del servidor: se pasan a |
//| UTC con el desfase actual (Exness no cambia de horario).         |
//+------------------------------------------------------------------+
bool Exportar()
  {
   const datetime ahora = TimeTradeServer();
   const long desfase = (long)MathRound((double)(ahora - TimeGMT()) / 1800.0) * 1800;
   const int archivo = FileOpen(TEMPORAL, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON, 0, CP_UTF8);
   if(archivo == INVALID_HANDLE)
     {
      Print("No se pudo abrir ", TEMPORAL, ": error ", GetLastError());
      return false;
     }
   FileWrite(archivo, "# exportado=" + IntegerToString((long)TimeGMT()));
   FileWrite(archivo, "time_utc\tmoneda\timportancia\tevento_id\tnombre");

   string monedas[];
   const int n_monedas = StringSplit(Monedas, ',', monedas);
   int escritos = 0;
   for(int m = 0; m < n_monedas; m++)
     {
      string moneda = monedas[m];
      StringTrimLeft(moneda);
      StringTrimRight(moneda);
      MqlCalendarValue valores[];
      ResetLastError();
      const int n = CalendarValueHistory(valores, ahora - (datetime)DiasDeHistoria * 86400,
                                         ahora + (datetime)DiasFuturos * 86400, NULL, moneda);
      if(n <= 0)
        {
         Print("Sin eventos de ", moneda, ": error ", GetLastError());
         continue;
        }
      for(int i = 0; i < n; i++)
        {
         MqlCalendarEvent evento;
         if(!CalendarEventById(valores[i].event_id, evento))
            continue;
         if((int)evento.importance < ImportanciaMinima || evento.time_mode != CALENDAR_TIMEMODE_DATETIME)
            continue; // sin hora exacta (todo el día, provisional) no sirve para el filtro
         string nombre = evento.name;
         StringReplace(nombre, "\t", " ");
         FileWrite(archivo, IntegerToString((long)valores[i].time - desfase) + "\t" + moneda + "\t" +
                   IntegerToString((int)evento.importance) + "\t" + IntegerToString((long)evento.id) + "\t" + nombre);
         escritos++;
        }
     }
   FileClose(archivo);
   if(escritos == 0)
     {
      Print("No se exportó ningún evento: el archivo anterior se conserva");
      FileDelete(TEMPORAL, FILE_COMMON);
      return false;
     }
   // El bot nunca ve un archivo a medias: se escribe aparte y se mueve.
   if(!FileMove(TEMPORAL, FILE_COMMON, ARCHIVO, FILE_COMMON | FILE_REWRITE))
     {
      Print("No se pudo reemplazar ", ARCHIVO, ": error ", GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
